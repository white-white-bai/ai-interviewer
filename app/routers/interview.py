"""面试会话接口：把状态机服务暴露为 HTTP API"""

from fastapi import APIRouter, HTTPException

from app.interview import service
from app.interview.schemas import (
    InterviewAnswerRequest,
    InterviewSession,
    InterviewStartRequest,
)
from app.interview.state_machine import InvalidTransitionError

# tags 在 /docs 文档中单独分组展示
router = APIRouter(prefix="/interview", tags=["AI 面试"])


@router.post("/start", response_model=InterviewSession, status_code=201)
async def start_interview(req: InterviewStartRequest):
    """开始一场面试：创建会话并生成第一道面试题

    LLM 未配置等环境问题返回 503 Service Unavailable。
    """
    try:
        return service.start_interview(req.candidate_name, req.position)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/{session_id}", response_model=InterviewSession)
async def get_interview(session_id: str):
    """查看面试会话当前状态与问答历史"""
    try:
        return service.get_session(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{session_id}/answer", response_model=InterviewSession)
async def submit_answer(session_id: str, req: InterviewAnswerRequest):
    """提交回答：AI 评估后决定追问 / 下一题 / 结束

    对已结束或不存在会话的非法操作分别返回 404 / 409。
    """
    try:
        return service.submit_answer(session_id, req.answer)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except InvalidTransitionError as exc:
        # 状态机拒绝的非法流转（如已 FINISHED 再答题）→ 409 Conflict
        raise HTTPException(status_code=409, detail=str(exc)) from exc
