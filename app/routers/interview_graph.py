"""LangGraph 版面试接口：与 M1 手写状态机接口并存，便于对比体验

对比 app/routers/interview.py：
- 不再需要 service 里的 _sessions 内存字典，状态全部来自检查点
- "提交回答"从普通函数调用变成 Command(resume=...) 断点恢复
"""

import uuid

from fastapi import APIRouter, HTTPException
from langgraph.types import Command

from app.interview.graph import interview_graph, make_config
from app.interview.schemas import (
    GraphInterviewSnapshot,
    InterviewAnswerRequest,
    InterviewApprovalRequest,
    InterviewStartRequest,
    QAItem,
)

router = APIRouter(prefix="/interview/graph", tags=["AI 面试 · LangGraph 版"])


def _snapshot(thread_id: str) -> GraphInterviewSnapshot:
    """从检查点读取会话快照：get_state 是 LangGraph 的"读档"API

    snap.next 是"接下来还要执行哪些节点"：非空说明图暂停在
    某个 interrupt 上（等回答或等审批），为空说明已跑到 END（面试结束）。
    """
    snap = interview_graph.get_state(make_config(thread_id))
    if not snap.values:
        raise HTTPException(status_code=404, detail=f"会话 {thread_id} 不存在")
    values = snap.values
    # 图可能暂停在不同含义的断点上（M8 起不止一个 interrupt）：
    # ask_candidate=等候选人回答，approval=等人工审批，空=已到 END
    if not snap.next:
        status = "finished"
    elif snap.next[0] == "approval":
        status = "waiting_approval"
    else:
        status = "waiting_answer"
    return GraphInterviewSnapshot(
        thread_id=thread_id,
        status=status,
        next_node=list(snap.next),
        candidate_name=values.get("candidate_name"),
        position=values.get("position"),
        current_question=values.get("current_question"),
        last_score=values.get("last_score"),
        last_comment=values.get("last_comment"),
        action=values.get("action"),
        summary=values.get("summary"),
        followup_count=values.get("followup_count", 0),
        round_count=values.get("round_count", 0),
        history=[QAItem(**qa) for qa in values.get("history", [])],
        memory_summary=values.get("memory_summary"),
    )


@router.post("/start", response_model=GraphInterviewSnapshot, status_code=201)
async def start(req: InterviewStartRequest):
    """开始面试：图跑到 interrupt 处自动暂停，返回第一道题

    固定路径注册在参数路径之前（工程规范）。
    """
    thread_id = uuid.uuid4().hex[:8]
    try:
        # 只传初始字段，图自己跑到 ask_candidate 的 interrupt() 暂停
        interview_graph.invoke(
            {"candidate_name": req.candidate_name, "position": req.position},
            make_config(thread_id),
        )
    except RuntimeError as exc:
        # LLM 未配置等环境问题 → 503 Service Unavailable
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return _snapshot(thread_id)


@router.get("/{thread_id}", response_model=GraphInterviewSnapshot)
async def get_interview(thread_id: str):
    """查看会话快照：数据来自检查点文件，服务重启后依然能读出完整状态"""
    return _snapshot(thread_id)


@router.post("/{thread_id}/answer", response_model=GraphInterviewSnapshot)
async def submit_answer(thread_id: str, req: InterviewAnswerRequest):
    """提交回答：Command(resume=...) 从断点恢复执行

    图会一口气跑完 评估 →（条件边）→ 出下一题 → 再次暂停，
    或 评估 → 总结 → END，全程无需外部干预。
    """
    snap = interview_graph.get_state(make_config(thread_id))
    if not snap.values:
        raise HTTPException(status_code=404, detail=f"会话 {thread_id} 不存在")
    if not snap.next:
        # 图已到 END，没有等待恢复的 interrupt → 409 Conflict
        raise HTTPException(status_code=409, detail="面试已结束，不能再提交回答")
    if snap.next[0] == "approval":
        # 防误用守卫（M8）：当前断点是审批而非答题——若把回答文本 resume 进
        # 审批 interrupt，bool(非空字符串) 恒为 True，等于"蒙混批准"
        raise HTTPException(
            status_code=409,
            detail="当前等待人工审批而非回答，请调用 /approval 接口",
        )
    try:
        interview_graph.invoke(Command(resume=req.answer), make_config(thread_id))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return _snapshot(thread_id)


@router.post("/{thread_id}/approval", response_model=GraphInterviewSnapshot)
async def submit_approval(thread_id: str, req: InterviewApprovalRequest):
    """人工审批（M8）：图暂停在 approval 节点时，用批准/驳回恢复执行

    批准 → summarize 生成终评 → END；驳回 → 回 plan_topic 继续面试。
    """
    snap = interview_graph.get_state(make_config(thread_id))
    if not snap.values:
        raise HTTPException(status_code=404, detail=f"会话 {thread_id} 不存在")
    if not snap.next:
        raise HTTPException(status_code=409, detail="面试已结束，无需审批")
    if snap.next[0] != "approval":
        raise HTTPException(
            status_code=409,
            detail="当前没有等待审批的决策（图暂停在其他断点）",
        )
    try:
        # resume 的值就是 approval 节点里 interrupt() 的返回值（True=批准 / False=驳回）
        interview_graph.invoke(Command(resume=req.approved), make_config(thread_id))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return _snapshot(thread_id)
