"""M9 演示：自我纠错——错误不是终点，而是反馈信号

三个场景（①已随 M8 修复落地，本脚本串演）：
- 幕一（场景①）：结构化输出坏 JSON → 把坏输出+错误原因回填给模型重试。
  用桩函数（monkeypatch llm.chat）确定性触发，观察 chat_structured 的重试日志。
- 幕二（场景②）：工具返回 error JSON → Agent 读到错误后下一轮自行修正/重试。
  用 monkeypatch 让 execute_tool 第一次返回超时错误，之后走真实现——
  模型的"重试决策"是真实 LLM 行为。
- 幕三（场景③）：<tool_call> 文本伪装工具调用（M3-b 伏笔兑现）→
  extract_text_tool_calls 提取 → 代为执行；顺带验证越权伪装仍被 M7 权限层拦截。

核心思想：可靠的 Agent 不是不犯错，而是"错了能看见、能反馈、能重来"——
同时重试必须有上限（熔断雏形，M11 系统化）。
"""

import json
import sys
from pathlib import Path

# 把项目根目录加入模块搜索路径（Windows 直接运行脚本时 sys.path 不含根目录）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import llm
from app.interview import tools
from app.interview.schemas import PlannerDirective
from app.interview.tools import extract_text_tool_calls, execute_tool, TOOLS


def act1_json_retry() -> None:
    """幕一：结构化输出坏 JSON → 错误回填 → 自动重试（场景①，桩函数确定性演示）

    monkeypatch 把 llm.chat 换成桩：第 1 次返回缺右括号的坏 JSON，
    第 2 次返回合法 JSON——模拟 DeepSeek 偶发的坏输出。
    chat_structured 应该：解析失败 → 打印纠错日志 → 回填重试 → 成功返回。
    """
    print("===== 幕一：结构化输出坏 JSON → 回填重试 =====")
    real_chat = llm.chat  # 保留真实现，演示完还原
    calls = {"n": 0}

    def flaky_chat(*, messages, temperature):
        """桩函数：第 1 次返回坏 JSON（缺右括号），第 2 次返回合法 JSON"""
        calls["n"] += 1
        if calls["n"] == 1:
            return '{"topic": "GIL", "guidance": "围绕 GIL 的释放时机追问"'  # 坏 JSON
        return '{"topic": "GIL 与多线程", "guidance": "围绕 GIL 释放时机与线程安全追问"}'

    llm.chat = flaky_chat  # 猴子补丁：替换 LLM 层
    try:
        directive = llm.chat_structured(
            messages=[{"role": "user", "content": "规划下一个考察点"}],
            model_cls=PlannerDirective,
        )
    finally:
        llm.chat = real_chat  # 无论成败都还原，避免污染其他演示
    print(f"共调用 LLM {calls['n']} 次（1 次失败 + {calls['n'] - 1} 次重试成功）")
    print(f"最终结果：topic={directive.topic} | guidance={directive.guidance}")


def act2_tool_error_retry() -> None:
    """幕二：工具返回 error JSON → Agent 自行重试（场景②，真实 LLM 决策）

    mini Agent 循环：声明 TOOLS → 模型发 tool_calls → 执行 → 回填。
    monkeypatch 让 execute_tool 第一次返回超时错误（模拟外部依赖抖动），
    之后走真实现——观察模型收到 error 后是否自行重试同样的调用。
    """
    print("\n\n===== 幕二：工具报错 → 模型自行重试 =====")
    real_execute = tools.execute_tool
    state = {"failed": False, "retried": False}

    def flaky_execute(name, arguments):
        """第一次调用返回超时 error，之后走真实现（模拟外部依赖抖动一次）"""
        if not state["failed"]:
            state["failed"] = True
            return json.dumps({"error": "题库连接超时，请稍后重试"}, ensure_ascii=False)
        state["retried"] = True  # 第二次及以后被调用 = 模型带着错误信息重试了
        return real_execute(name, arguments)

    tools.execute_tool = flaky_execute
    try:
        messages = [
            {"role": "system", "content": "你是出题助手，先查题库再决定出什么题。"},
            {"role": "user", "content": "请查询题库中关键词为 GIL 的题目，告诉我查到几道。"},
        ]
        for round_no in range(1, 4):  # 最多 3 轮（与图内 MAX_TOOL_ROUNDS 同款熔断）
            message = llm.chat_with_tools(messages, tools=TOOLS, temperature=0.3)
            if not message.tool_calls:
                print(f"[第 {round_no} 轮·最终回答] {(message.content or '').strip()[:120]}")
                break
            messages.append(message)
            for call in message.tool_calls:
                print(f"[第 {round_no} 轮·工具调用] {call.function.name}({call.function.arguments})")
                result = tools.execute_tool(call.function.name, call.function.arguments)
                print(f"[第 {round_no} 轮·工具结果] {result[:120]}")
                messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
    finally:
        tools.execute_tool = real_execute  # 还原真实现
    if state["retried"]:
        print("观察点：注入超时 error 后，模型没有放弃，而是带着错误信息重试了调用——"
              "错误回填就是自我纠错的燃料。")
    else:
        print("观察点：本次模型未重试（LLM 有随机性，可重跑观察；图内同类场景有 MAX_TOOL_ROUNDS 熔断兜底）")


def act3_fake_tool_call() -> None:
    """幕三：<tool_call> 文本伪装 → 提取接管（场景③，确定性演示，零 LLM 调用）

    模拟 DeepSeek 把工具调用按训练模板写进正文的抽风输出：
    1. 合法工具的伪装调用 → 提取并真实执行（查题库）
    2. 越权工具的伪装调用 → 提取后仍被 M7 权限层拦截（M9 × M7 联动）
    """
    print("\n\n===== 幕三：<tool_call> 文本伪装 → 提取接管 =====")
    fake_output = (
        "我先查一下题库再出题。\n"
        '<tool_call>\n{"name": "search_question_bank", "arguments": {"keyword": "GIL"}}\n</tool_call>\n'
        "请稍等。"
    )
    print(f"模拟的抽风输出：{fake_output.strip()}")
    calls = extract_text_tool_calls(fake_output)
    print(f"\n提取到 {len(calls)} 个伪装调用（若无 M9 检测，这段会污染最终问题文本）")
    for call in calls:
        result = execute_tool(call["name"], call["arguments"])
        print(f"接管执行 → {result[:150]}")

    # 越权伪装：即使模型伪装成正文，也绕不过权限层
    evil = extract_text_tool_calls(
        '<tool_call>{"name": "drop_all_tables", "arguments": {}}</tool_call>'
    )
    print(f"\n伪装的越权调用提取：{evil}")
    print(f"接管执行 → {execute_tool(evil[0]['name'], evil[0]['arguments'])}")

    # 坏 JSON 伪装块：解析失败直接忽略，不影响其他块
    broken = extract_text_tool_calls(
        '<tool_call>{"name": "search_question_bank", "arguments": {broken</tool_call>'
    )
    print(f"坏 JSON 伪装块提取结果：{broken}（空列表 = 安全忽略）")


def main() -> None:
    """依次演示三个自我纠错场景"""
    act1_json_retry()
    act2_tool_error_retry()
    act3_fake_tool_call()
    print("\n===== M9 演示结束 =====")


if __name__ == "__main__":
    main()
