from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException

from app.dependencies import get_question_or_404, pagination_params
from app.models import Question, QuestionCreate, QuestionUpdate
from app.storage import fake_questions

# APIRouter：把一组相关路由组织成模块，由 main.py 的 include_router 挂载
# prefix：为本模块所有路径统一加前缀（列表路由路径写 ""，拼出 /questions）
# tags：在 /docs 文档中分组展示
router = APIRouter(prefix="/questions", tags=["面试题"])


@router.get("/search")
async def search_questions(
    pagination: tuple[int, int] = Depends(pagination_params),
    category: str = "",
    keyword: str = "",
):
    """按分类/关键词搜索面试题（支持分页）

    注意：固定路径 /questions/search 必须定义在参数路径
    /questions/{question_id} 之前，否则 "search" 会先被
    {question_id} 匹配导致 422 校验错误。
    """
    results = [
        q for q in fake_questions
        # 分类为空表示不过滤；关键词为空表示不过滤
        if (not category or q.category == category)
        and (not keyword or keyword in q.title)
    ]
    page, size = pagination
    start = (page - 1) * size
    return {"questions": results[start : start + size], "total": len(results)}


@router.get("")
async def list_questions(pagination: tuple[int, int] = Depends(pagination_params)):
    """获取面试题列表（支持分页：?page=1&size=10）"""
    page, size = pagination
    start = (page - 1) * size
    return {
        "questions": fake_questions[start : start + size],
        "total": len(fake_questions),
        "page": page,
        "size": size,
    }


@router.get("/{question_id}")
async def get_question(question: Question = Depends(get_question_or_404)):
    """根据 ID 获取单道面试题"""
    return question


@router.post("", response_model=Question, status_code=201)
async def create_question(data: QuestionCreate):
    """创建面试题

    请求体参数类型为 Pydantic 模型时，FastAPI 自动把 JSON 请求体
    解析并校验为该模型实例；response_model 限定响应结构。
    """
    # 自增 ID：当前最大 ID + 1
    new_id = max((q.id for q in fake_questions), default=0) + 1
    # model_dump() 把模型转为 dict，再用 ** 展开合并进完整 Question
    question = Question(id=new_id, created_at=datetime.now(), **data.model_dump())
    fake_questions.append(question)
    return question


@router.put("/{question_id}", response_model=Question)
async def update_question(
    data: QuestionUpdate,
    question: Question = Depends(get_question_or_404),
):
    """更新面试题（部分更新：只修改请求体中显式提供的字段）"""
    # exclude_unset=True：只取客户端显式传入的字段，
    # 未传字段保持原值，实现 PATCH 式的部分更新
    for field, value in data.model_dump(exclude_unset=True).items():
        setattr(question, field, value)
    return question


@router.delete("/{question_id}", status_code=204)
async def delete_question(question: Question = Depends(get_question_or_404)):
    """删除面试题：成功返回 204 No Content（无响应体）"""
    fake_questions.remove(question)
    return
