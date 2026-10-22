"""Agent 工具层（M3-b）：把业务能力封装成 LLM 可通过 function calling 调用的工具

工具 = 三部分组成：
1. TOOLS：JSON Schema 声明（告诉模型工具叫什么、干什么、参数是什么）
2. execute_tool：真正的执行器（模型只"点名"，干活的是我们的代码）
3. 回填协议：执行结果以 role=tool 消息追加回对话，模型据此继续生成

安全边界：模型只能"请求"调用工具，真正的执行权在我们的代码手里——
这是 M7（工具权限）要系统化的思想雏形。
"""

import json

from app import storage

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


def execute_tool(name: str, arguments: str) -> str:
    """执行一次工具调用，返回 JSON 字符串结果（用于回填给模型）

    Args:
        name: 工具名（模型从 TOOLS 里选的）
        arguments: 模型生成的 JSON 字符串参数，需要解析后使用

    Returns:
        序列化为 JSON 字符串的工具执行结果
    """
    # 模型输出的 arguments 偶尔会是空串，兜底解析为空对象
    args = json.loads(arguments) if arguments else {}
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
    return json.dumps({"error": f"未知工具：{name}"}, ensure_ascii=False)
