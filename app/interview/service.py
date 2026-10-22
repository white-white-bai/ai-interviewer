"""面试业务服务：组合状态机与 LLM 调用（M1 阶段内存存储，M3 换 SQLite）"""

import uuid
from datetime import datetime

from app import llm
from app.interview.prompts import EVALUATE_PROMPT, QUESTION_PROMPT
from app.interview.schemas import (
    EvaluationResult,
    InterviewSession,
    NextAction,
    QAItem,
)
from app.interview.state_machine import InterviewState, transition

# 内存会话存储：{session_id: InterviewSession}
_sessions: dict[str, InterviewSession] = {}


def _format_history(session: InterviewSession) -> str:
    """把历史问答整理成提示词文本；无历史时返回占位说明"""
    if not session.history:
        return "（暂无，这是第一个问题）"
    lines = []
    for i, qa in enumerate(session.history, 1):
        lines.append(f"{i}. 问：{qa.question}\n   答：{qa.answer}（{qa.score} 分）")
    return "\n".join(lines)


def _clean_json(text: str) -> str:
    """清洗 LLM 输出：去掉 markdown 代码围栏，得到纯 JSON 文本

    模型常无视指令输出 ```json ... ``` 围栏，这里做兜底清洗；
    更健壮的重试机制在 M9（自我纠错）实现。
    """
    text = text.strip()
    if text.startswith("```"):
        # 去掉首行（```json）与结尾的 ``` 围栏
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    return text.strip()


def start_interview(candidate_name: str, position: str) -> InterviewSession:
    """创建面试会话并生成第一道面试题

    状态流转：CREATED --start--> WAITING_ANSWER
    """
    session = InterviewSession(
        id=uuid.uuid4().hex[:8],
        candidate_name=candidate_name,
        position=position,
    )
    # 调用 LLM 生成第一题（出题用高温度，鼓励发散覆盖）
    question = llm.chat(
        messages=[{
            "role": "user",
            "content": QUESTION_PROMPT.format(
                position=position,
                candidate_name=candidate_name,
                history_text=_format_history(session),
            ),
        }],
        temperature=0.8,
    )
    session.current_question = question.strip()
    # 状态变化必须走转移表，不允许直接赋值 state 字段
    session.state = transition(session.state, "start")
    _sessions[session.id] = session
    return session


def submit_answer(session_id: str, answer: str) -> InterviewSession:
    """提交回答 → AI 评估 → 按评估结果流转（追问 / 下一题 / 结束）

    状态流转：WAITING_ANSWER --submit_answer--> EVALUATING
              EVALUATING --ask_next--> WAITING_ANSWER
              EVALUATING --finish--> FINISHED
    """
    session = _sessions.get(session_id)
    if session is None:
        raise KeyError(f"会话 {session_id} 不存在")

    # 先流转到"评估中"，LLM 调用若崩溃，能从状态看出死在哪一步
    session.state = transition(session.state, "submit_answer")

    # 评估用低温度（0.2），保证打分稳定
    raw = llm.chat(
        messages=[{
            "role": "user",
            "content": EVALUATE_PROMPT.format(
                position=session.position,
                question=session.current_question,
                answer=answer,
            ),
        }],
        temperature=0.2,
    )
    # 结构化解析：Pydantic 校验 score 范围与 action 枚举合法性
    result = EvaluationResult.model_validate_json(_clean_json(raw))

    # 记录本轮问答
    session.history.append(QAItem(
        question=session.current_question,
        answer=answer,
        score=result.score,
        comment=result.comment,
        created_at=datetime.now(),
    ))

    if result.action is NextAction.FINISH:
        # 结束面试，清空当前题
        session.state = transition(session.state, "finish")
        session.current_question = None
    else:
        if result.action is NextAction.FOLLOWUP:
            session.followup_count += 1
        # 追问与下一题都复用出题提示词（历史里已包含刚评估的问答）
        question = llm.chat(
            messages=[{
                "role": "user",
                "content": QUESTION_PROMPT.format(
                    position=session.position,
                    candidate_name=session.candidate_name,
                    history_text=_format_history(session),
                ),
            }],
            temperature=0.8,
        )
        session.current_question = question.strip()
        session.state = transition(session.state, "ask_next")
    return session


def get_session(session_id: str) -> InterviewSession:
    """按 ID 获取会话，不存在抛出 KeyError"""
    session = _sessions.get(session_id)
    if session is None:
        raise KeyError(f"会话 {session_id} 不存在")
    return session
