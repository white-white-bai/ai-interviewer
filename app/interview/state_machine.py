"""面试会话状态机：显式声明所有合法状态转移，杜绝"流浪状态"

为什么不用 while True 循环？
- while True：状态散落在循环变量里，崩溃后无法知道"死在哪一步"，也无法暂停恢复
- 状态机：所有流转登记在转移表里，非法事件直接抛异常，
  会话随时可序列化保存（M2 LangGraph 会把这套思路声明式重写）
"""

from enum import Enum


class InterviewState(str, Enum):
    """面试会话的所有可能状态"""

    CREATED = "created"                # 会话已创建，尚未开始
    WAITING_ANSWER = "waiting_answer"  # 已出题，等待候选人回答
    EVALUATING = "evaluating"          # 收到回答，AI 评估中
    FINISHED = "finished"              # 面试结束（终态）


class InvalidTransitionError(Exception):
    """非法状态转移异常：当前状态下发生了不该发生的事件"""


# 状态转移表：{当前状态: {事件名: 目标状态}}
# 设计要点：所有状态变化必须走这张表，代码里不允许直接改 state 字段
TRANSITIONS: dict[InterviewState, dict[str, InterviewState]] = {
    InterviewState.CREATED: {
        "start": InterviewState.WAITING_ANSWER,
    },
    InterviewState.WAITING_ANSWER: {
        "submit_answer": InterviewState.EVALUATING,
    },
    InterviewState.EVALUATING: {
        "ask_next": InterviewState.WAITING_ANSWER,  # 追问或下一题
        "finish": InterviewState.FINISHED,          # 评估后决定结束
    },
    InterviewState.FINISHED: {},  # 终态：不允许任何事件（这是刻意设计）
}


def transition(current: InterviewState, event: str) -> InterviewState:
    """执行状态转移

    Args:
        current: 当前状态
        event: 触发转移的事件名

    Returns:
        转移后的新状态

    Raises:
        InvalidTransitionError: 当前状态下该事件不合法时抛出
    """
    event_map = TRANSITIONS.get(current, {})
    if event not in event_map:
        raise InvalidTransitionError(
            f"状态 {current.value} 不允许事件 {event}，合法事件：{list(event_map)}"
        )
    return event_map[event]
