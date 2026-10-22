from fastapi import APIRouter, Depends, HTTPException

from app import storage
from app.dependencies import get_question_or_404, pagination_params
from app.models import Question, QuestionCreate, QuestionUpdate

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

    M3 变化：数据操作全部委托给 storage 仓储层，路由不再直接碰数据。
    """
    results = storage.search_questions(category=category, keyword=keyword)
    page, size = pagination
    start = (page - 1) * size
    return {"questions": results[start : start + size], "total": len(results)}


@router.get("")
async def list_questions(pagination: tuple[int, int] = Depends(pagination_params)):
    """获取面试题列表（支持分页：?page=1&size=10）"""
    page, size = pagination
    items, total = storage.list_questions(page=page, size=size)
    return {"questions": items, "total": total, "page": page, "size": size}


@router.get("/{question_id}")
async def get_question(question: Question = Depends(get_question_or_404)):
    """根据 ID 获取单道面试题"""
    return question


@router.post("", response_model=Question, status_code=201)
async def create_question(data: QuestionCreate):
    """创建面试题（M3 起由 SQLite 自增主键生成 id）"""
    return storage.create_question(data)


@router.put("/{question_id}", response_model=Question)
async def update_question(
    data: QuestionUpdate,
    question: Question = Depends(get_question_or_404),
):
    """更新面试题（部分更新：只修改请求体中显式提供的字段）"""
    return storage.update_question(question.id, data)


@router.delete("/{question_id}", status_code=204)
async def delete_question(question: Question = Depends(get_question_or_404)):
    """删除面试题：成功返回 204 No Content（无响应体）"""
    storage.delete_question(question.id)
    return
