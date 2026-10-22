"""M2：用 LangGraph 重写面试流程（与 M1 手写状态机 app/interview/state_machine.py 并存对比）

M1 的痛点 → M2 的解法：
- 状态存在内存字典里，重启即丢 → checkpointer 每执行一步自动把完整状态写入 SQLite
- 追问/下一题/结束的分支写在 if/else 里 → add_conditional_edges 声明式注册
- 没有"停在这里等外部输入"的机制 → interrupt() 官方人工介入点

图的拓扑（一次完整面试的执行轨迹）：
START → generate_question（出题）
      → ask_candidate（interrupt 暂停，等候选人提交答案）
      → evaluate（评估打分）
      → 条件边：followup/next → 回到 generate_question（循环）
               finish        → summarize → END
"""

import operator
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Annotated, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from app import llm
from app.interview.prompts import EVALUATE_PROMPT, QUESTION_PROMPT, SUMMARY_PROMPT
from app.interview.schemas import EvaluationResult, NextAction, QAItem
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


def _format_history(history: list[dict]) -> str:
    """把历史问答（QAItem 字典列表）整理成提示词文本；无历史时返回占位说明"""
    if not history:
        return "（暂无，这是第一个问题）"
    lines = []
    for i, qa in enumerate(history, 1):
        lines.append(f"{i}. 问：{qa['question']}\n   答：{qa['answer']}（{qa['score']} 分）")
    return "\n".join(lines)


def generate_question(state: InterviewGraphState) -> dict:
    """出题节点：根据岗位 + 历史问答生成下一道面试题

    出题用高温度（0.8），鼓励发散覆盖更多知识点。
    返回 dict 是"增量更新"：LangGraph 会把它合并进共享状态。
    """
    question = llm.chat(
        messages=[{
            "role": "user",
            "content": QUESTION_PROMPT.format(
                position=state["position"],
                candidate_name=state["candidate_name"],
                history_text=_format_history(state.get("history", [])),
            ),
        }],
        temperature=0.8,
    )
    return {
        "current_question": question.strip(),
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


def evaluate(state: InterviewGraphState) -> dict:
    """评估节点：低温度（0.2）打分 + Pydantic 结构化校验 + 记录本轮问答"""
    raw = llm.chat(
        messages=[{
            "role": "user",
            "content": EVALUATE_PROMPT.format(
                position=state["position"],
                question=state["current_question"],
                answer=state["last_answer"],
            ),
        }],
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
    builder.add_node("generate_question", generate_question)
    builder.add_node("ask_candidate", ask_candidate)
    builder.add_node("evaluate", evaluate)
    builder.add_node("summarize", summarize)

    builder.add_edge(START, "generate_question")
    builder.add_edge("generate_question", "ask_candidate")
    builder.add_edge("ask_candidate", "evaluate")
    # 条件边：路由函数返回值 → 映射表 → 下一个节点
    # followup 和 next 都回到出题节点，区别只体现在下一题的提示词历史里
    builder.add_conditional_edges(
        "evaluate",
        route_after_evaluate,
        {"followup": "generate_question", "next": "generate_question", "finish": "summarize"},
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
