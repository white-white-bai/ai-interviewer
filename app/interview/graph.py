"""M2：用 LangGraph 重写面试流程（与 M1 手写状态机 app/interview/state_machine.py 并存对比）

M1 的痛点 → M2 的解法：
- 状态存在内存字典里，重启即丢 → checkpointer 每执行一步自动把完整状态写入 SQLite
- 追问/下一题/结束的分支写在 if/else 里 → add_conditional_edges 声明式注册
- 没有"停在这里等外部输入"的机制 → interrupt() 官方人工介入点

图的拓扑（一次完整面试的执行轨迹）：
START → plan_topic（M5：规划官决定下一个考察方向）
      → generate_question（M5：面试官按规划指令定向出题）
      → ask_candidate（interrupt 暂停，等候选人提交答案）
      → evaluate（M5：评估官独立人设打分）
      → manage_memory（M4：滑窗外的早期问答滚动压缩进摘要）
      → 条件边：followup/next → 回到 plan_topic（循环）
               finish        → summarize → END

三个 Agent（规划官/面试官/评估官）不互相喊话，
通过共享 state 的 directive 字段传递结构化消息。
"""

import operator
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Annotated, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from app import llm, sandbox
from app.interview import agents
from app.interview.memory import format_memory, manage_memory
from app.interview.prompts import (
    EVALUATE_PROMPT,
    QUESTION_WITH_TOOLS_PROMPT,
    SUMMARY_PROMPT,
)
from app.interview.schemas import EvaluationResult, NextAction, QAItem
from app.interview.tools import TOOLS, execute_tool
from app.interview.service import _clean_json

# 检查点数据库：LangGraph 每个步骤都会把状态快照写进这个文件
# 注意：这是 LangGraph 自己的"存档库"，与 M3 要做的业务数据库是两回事
_CHECKPOINT_DB = Path(__file__).resolve().parent.parent.parent / "checkpoints.sqlite"

# FastAPI 多线程处理请求，sqlite3 连接必须允许跨线程使用
_conn = sqlite3.connect(_CHECKPOINT_DB, check_same_thread=False)
checkpointer = SqliteSaver(_conn)


class InterviewGraphState(TypedDict):
    """图的共享状态：所有节点读写同一份 state，LangGraph 负责合并与存档

    Annotated[list[dict], operator.add] 是"reducer"声明：
    节点返回 {"history": [新一条]} 时，LangGraph 用列表相加合并而不是覆盖——
    这就是历史问答自动累加的秘密（M1 里要手动 append）。
    """

    candidate_name: str
    position: str
    current_question: str | None
    last_answer: str | None
    last_score: int | None
    last_comment: str | None
    action: str | None
    history: Annotated[list[dict], operator.add]
    followup_count: int
    round_count: int
    summary: str | None
    # M4 记忆字段：摘要 + 进度水位线（前 memory_upto 轮已并入摘要）
    memory_summary: str | None
    memory_upto: int
    # M5 多 Agent 字段：规划官写给面试官的定向出题指令（PlannerDirective 字典）
    directive: dict | None
    # M6 沙箱字段：候选人代码块的真实运行结果（本轮没有代码块则为 None）
    code_result: str | None


def _format_history(history: list[dict]) -> str:
    """全量拼接历史问答（M4 起仅供 summarize 终评使用——总结需要完整信息）

    出题提示词已改用 memory.format_memory（摘要 + 滑窗），
    这里保留是因为最终评语必须覆盖每一轮，属于一次性开销。
    """
    if not history:
        return "（暂无，这是第一个问题）"
    lines = []
    for i, qa in enumerate(history, 1):
        lines.append(f"{i}. 问：{qa['question']}\n   答：{qa['answer']}（{qa['score']} 分）")
    return "\n".join(lines)


# Agent 循环最大工具轮数：防止模型无限调工具烧钱（成本熔断雏形，M11 系统化）
MAX_TOOL_ROUNDS = 3


def plan_topic(state: InterviewGraphState) -> dict:
    """考察规划节点（M5 新增）：多 Agent 协作的第一环

    规划官读记忆文本（摘要+滑窗），决定下一个考察知识点，
    结果经 Pydantic 校验后写入 state["directive"]——
    Agent 之间不直接对话，图的共享状态就是协作介质。
    """
    directive = agents.plan_next_topic(state)
    print(f"[规划官] 下一考点：{directive.topic}（{directive.guidance}）")
    return {"directive": directive.model_dump()}


def generate_question(state: InterviewGraphState) -> dict:
    """出题节点（M5 升级为面试官 Agent）：按规划官指令定向出题

    Agent 循环（function calling 标准套路，M3-b）：
    1. 工具声明随消息发给模型
    2. 模型要么直接给出问题文本（循环结束），要么返回 tool_calls
    3. 执行工具，结果以 role=tool 消息回填，回到第 1 步

    M5 变化：messages[0] 换成面试官角色的 system prompt，
    user 消息里注入规划官的 directive——出题从"自由发挥"变"按计划执行"。

    出题用高温度（0.8），鼓励发散覆盖更多知识点。
    """
    directive = state.get("directive") or {}
    directive_text = (
        f"知识点：{directive.get('topic', '自由选择')}；"
        f"指令：{directive.get('guidance', '无')}"
    )
    user_content = QUESTION_WITH_TOOLS_PROMPT.format(
        position=state["position"],
        candidate_name=state["candidate_name"],
        directive=directive_text,
        memory_text=format_memory(state),
    )
    messages: list = [
        {"role": "system", "content": agents.INTERVIEWER_SYSTEM.format(position=state["position"])},
        {"role": "user", "content": user_content},
    ]
    question = ""
    for _ in range(MAX_TOOL_ROUNDS):
        message = llm.chat_with_tools(messages, tools=TOOLS, temperature=0.8)
        if not message.tool_calls:
            # 模型不再要工具了：message.content 就是最终问题
            question = (message.content or "").strip()
            break
        # 协议要求：先把 assistant 的 tool_calls 消息入列，再回填对应结果
        messages.append(message)
        for call in message.tool_calls:
            # print 是教学期的轨迹观察手段，M10 会换成正式的轨迹记录
            print(f"[面试官·工具调用] {call.function.name}({call.function.arguments})")
            result = execute_tool(call.function.name, call.function.arguments)
            print(f"[面试官·工具结果] {result[:200]}")
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,  # 结果必须与请求一一对应
                "content": result,
            })
    if not question:
        # 兜底：达到工具轮数上限仍没拿到最终问题，撤掉工具声明直接要答案
        # 注意 M5 后 messages[0] 是 system 人设，必须保留 [:2] 再撤工具
        question = llm.chat(messages=messages[:2], temperature=0.8).strip()
    return {
        "current_question": question,
        "round_count": state.get("round_count", 0) + 1,
    }


def ask_candidate(state: InterviewGraphState) -> dict:
    """人工介入节点：interrupt() 把图"暂停"在这里（M2 核心机制）

    - 首次执行到 interrupt()：图立刻停下，当前状态存入检查点，
      payload（当前题目）可通过 get_state 读出，进程甚至可以直接退出
    - 外部调用 invoke(Command(resume=答案)) 后：图从这个断点"原地复活"，
      interrupt() 的返回值就是候选人提交的答案
    """
    answer = interrupt({"question": state["current_question"]})
    return {"last_answer": answer}


# 代码块提取：匹配 ```python ... ``` 或无语言标注的 ``` ... ```（DOTALL 跨行）
_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def run_code(state: InterviewGraphState) -> dict:
    """代码沙箱节点（M6 新增）：候选人回答里的代码块真正跑一遍

    Agent 决策依据的升级：评估官不再猜"这段代码能不能跑"，
    而是拿到真实运行结果（输出/报错/超时）再打分——
    LLM 心算运行结果经常出错，沙箱给的是事实。

    本轮回答没有代码块时显式把 code_result 置回 None：
    该字段留在 state 里会跨轮残留，必须每轮清理。
    """
    answer = state.get("last_answer") or ""
    matches = list(_CODE_BLOCK_RE.finditer(answer))
    if not matches:
        return {"code_result": None}  # 无代码块：清理旧值并空转通过
    # 取最后一个代码块：候选人常先用小片段讲解，最终答案代码通常放在最后
    code = matches[-1].group(1)
    print(f"[沙箱] 检测到 {len(matches)} 个代码块，执行最后一个（超时 {sandbox.DEFAULT_TIMEOUT} 秒）...")
    result = sandbox.run_python(code)
    label = "成功" if result["ok"] else ("超时强杀" if result["timed_out"] else "报错")
    preview = (result["stdout"] or result["stderr"]).strip()[:120]
    print(f"[沙箱] 执行{label}：{preview}")
    return {"code_result": sandbox.format_result(result)}


def evaluate(state: InterviewGraphState) -> dict:
    """评估节点（M5 升级为评估官 Agent）：低温度（0.2）打分 + Pydantic 结构化校验 + 记录本轮问答

    评估官有独立的 system 人设（agents.EVALUATOR_SYSTEM），
    与出题的面试官分离——避免"出题者自己评估自己"的角色混淆。
    M6：若沙箱跑出了真实运行结果，作为证据追加进评估输入——
    不改共享的 EVALUATE_PROMPT 模板（M1 的 service 也在用它）。
    """
    answer = state["last_answer"]
    if state.get("code_result"):
        answer = (
            f"{answer}\n\n"
            f"【沙箱实测】候选人代码块的真实运行结果（以此为准，不要凭记忆猜测代码输出）：\n"
            f"{state['code_result']}"
        )
    raw = llm.chat(
        messages=[
            {"role": "system", "content": agents.EVALUATOR_SYSTEM},
            {
                "role": "user",
                "content": EVALUATE_PROMPT.format(
                    position=state["position"],
                    question=state["current_question"],
                    answer=state["last_answer"],
                ),
            },
        ],
        temperature=0.2,
    )
    result = EvaluationResult.model_validate_json(_clean_json(raw))
    qa = QAItem(
        question=state["current_question"],
        answer=state["last_answer"],
        score=result.score,
        comment=result.comment,
        created_at=datetime.now(),
    )
    updates: dict = {
        "last_score": result.score,
        "last_comment": result.comment,
        "action": result.action.value,
        "history": [qa.model_dump()],  # history 声明了 operator.add，这里只给增量
    }
    if result.action is NextAction.FOLLOWUP:
        updates["followup_count"] = state.get("followup_count", 0) + 1
    return updates


def route_after_evaluate(state: InterviewGraphState) -> str:
    """条件边路由函数：返回值决定走哪条边

    返回值必须与 add_conditional_edges 注册的映射键一致
    （"followup" / "next" / "finish"），相当于 M1 转移表的声明式版本。
    """
    return state["action"]


def summarize(state: InterviewGraphState) -> dict:
    """总结节点（finish 分支专属）：生成整场面试的评语，随后图走到 END"""
    report = llm.chat(
        messages=[{
            "role": "user",
            "content": SUMMARY_PROMPT.format(
                position=state["position"],
                candidate_name=state["candidate_name"],
                history_text=_format_history(state.get("history", [])),
            ),
        }],
        temperature=0.4,
    )
    return {"summary": report.strip(), "current_question": None}


def build_interview_graph():
    """组装并编译面试图：节点 + 边 + 条件边 + 检查点持久化

    add_edge 是固定流转（等价于 M1 转移表里的一行），
    add_conditional_edges 是"看状态选路"（等价于 M1 里的 if/else 分支）。
    """
    builder = StateGraph(InterviewGraphState)
    builder.add_node("plan_topic", plan_topic)  # M5：规划官 Agent
    builder.add_node("generate_question", generate_question)
    builder.add_node("ask_candidate", ask_candidate)
    builder.add_node("run_code", run_code)  # M6：代码沙箱节点
    builder.add_node("evaluate", evaluate)
    builder.add_node("manage_memory", manage_memory)  # M4：记忆管理节点
    builder.add_node("summarize", summarize)

    builder.add_edge(START, "plan_topic")
    builder.add_edge("plan_topic", "generate_question")
    builder.add_edge("generate_question", "ask_candidate")
    builder.add_edge("ask_candidate", "run_code")
    builder.add_edge("run_code", "evaluate")
    builder.add_edge("evaluate", "manage_memory")
    # 条件边从 manage_memory 挂出：history 已更新完毕，摘要也已滚动，
    # followup/next 都回到规划官——每一道新题都先经规划再出题
    builder.add_conditional_edges(
        "manage_memory",
        route_after_evaluate,
        {"followup": "plan_topic", "next": "plan_topic", "finish": "summarize"},
    )
    builder.add_edge("summarize", END)
    # compile 时挂上 checkpointer：之后每一步自动写档，支持暂停/恢复/读档
    return builder.compile(checkpointer=checkpointer)


# 模块级单例：应用生命周期内复用同一个图和同一个检查点连接
interview_graph = build_interview_graph()


def make_config(thread_id: str) -> dict:
    """构造 LangGraph 运行配置

    thread_id 是检查点里的"会话主键"（对应 M1 的 session_id）：
    同一 thread_id 的所有调用共享并续写同一份存档。
    """
    return {"configurable": {"thread_id": thread_id}}
