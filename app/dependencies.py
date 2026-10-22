from fastapi import HTTPException, Query

from app.models import Question
from app.storage import fake_questions


def get_question_or_404(question_id: int) -> Question:
    """依赖函数：按 ID 查找题目，找不到抛出 404

    依赖函数的参数同样由 FastAPI 自动解析——question_id
    会自动从路径参数 {question_id} 注入，无需手动传值。
    """
    for q in fake_questions:
        if q.id == question_id:
            return q
    raise HTTPException(status_code=404, detail=f"题目 {question_id} 不存在")


def pagination_params(
    page: int = Query(1, ge=1, description="页码，从 1 开始"),
    size: int = Query(10, ge=1, le=100, description="每页条数"),
) -> tuple[int, int]:
    """依赖函数：通用分页参数，返回 (page, size)

    多个列表类端点复用同一套分页规则与校验，避免重复声明。
    """
    return page, size
