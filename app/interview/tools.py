"""Agent 工具层（M3-b）：把业务能力封装成 LLM 可通过 function calling 调用的工具

工具 = 三部分组成：
1. TOOLS：JSON Schema 声明（告诉模型工具叫什么、干什么、参数是什么）
2. execute_tool：真正的执行器（模型只"点名"，干活的是我们的代码）
3. 回填协议：执行结果以 role=tool 消息追加回对话，模型据此继续生成

安全边界（M7 落地）：模型只能"请求"调用工具，真正的执行权在我们的代码手里——
execute_tool 执行前先过 security.check_tool_permission（白名单校验 + 参数清洗）。
"""

import json
import re

from app import security, storage

# 工具声明（OpenAI function calling 格式，DeepSeek 兼容）
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_question_bank",
            "description": (
                "查询面试题库，按分类和关键词检索已有题目。"
                "出题前建议先查询：优先复用题库中的题目，查不到合适的再原创。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "题目关键词，如 GIL、索引、事务",
                    },
                    "category": {
                        "type": "string",
                        "description": "题目分类，如 Python、数据库、网络",
                    },
                },
                "required": [],
            },
        },
    }
]


# <tool_call> 文本伪装块的提取正则（M9 场景③）：匹配模型按训练模板格式
# 写在正文里的工具调用，非贪婪跨行匹配花括号内容
_TEXT_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def extract_text_tool_calls(text: str) -> list[dict]:
    """从模型正文中提取 <tool_call> 标签伪装的工具调用（M9 场景③）

    背景（M3-b 伏笔兑现）：DeepSeek 偶发把工具调用按训练模板格式写成
    正文文本 <tool_call>{...}</tool_call>，OpenAI 兼容层解析不出
    tool_calls 字段——伪装文本会被当成最终回答输出，Agent 循环就此漏掉
    一次该做的工具查询。

    处理策略：解析出 name/arguments 交给调用方（graph.generate_question）
    代为执行；单个块解析失败（坏 JSON）直接忽略，不影响其他块。

    Args:
        text: 模型输出的正文文本

    Returns:
        [{"name": 工具名, "arguments": JSON 字符串参数}]，无伪装块时为空列表
    """
    calls: list[dict] = []
    for match in _TEXT_TOOL_CALL_RE.finditer(text):
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue  # 伪装块本身也是坏 JSON：忽略此块，不中断其他块
        name = payload.get("name", "")
        args = payload.get("arguments", {})
        # arguments 可能是对象也可能是字符串，统一归一成 JSON 字符串（execute_tool 的入参约定）
        arguments = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
        if name:
            calls.append({"name": name, "arguments": arguments})
    return calls


def execute_tool(name: str, arguments: str) -> str:
    """执行一次工具调用，返回 JSON 字符串结果（用于回填给模型）

    M7：执行前先过权限层——白名单校验 + 参数清洗。
    模型只能"请求"工具，执行权永远在我们手里：
    越权请求不执行、不崩溃，返回结构化拒绝信息回填给模型。

    Args:
        name: 工具名（模型从 TOOLS 里选的）
        arguments: 模型生成的 JSON 字符串参数，需要解析后使用

    Returns:
        序列化为 JSON 字符串的工具执行结果（或拒绝原因）
    """
    # 权限关卡：不在白名单 / 参数畸形 → 直接回填拒绝信息
    allowed, denial, args = security.check_tool_permission(name, arguments)
    if not allowed:
        return denial
    if name == "search_question_bank":
        results = storage.search_questions(
            category=args.get("category", ""),
            keyword=args.get("keyword", ""),
        )
        payload = [{"id": q.id, "title": q.title, "category": q.category} for q in results]
        return json.dumps(
            {"count": len(payload), "questions": payload}, ensure_ascii=False
        )
    # 未知工具：返回错误信息而不是抛异常，让模型有机会自行纠正（M9 主题）
    # （绝大多数未知工具已被白名单拦截，这里兜底"在名单但未实现"的情况）
    return json.dumps({"error": f"未知工具：{name}"}, ensure_ascii=False)
