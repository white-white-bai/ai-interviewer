from datetime import datetime

from pydantic import BaseModel, Field


class QuestionCreate(BaseModel):
    """创建面试题的请求体模型：声明客户端可提交的字段与校验规则"""

    title: str = Field(min_length=5, max_length=200, description="题目内容")
    category: str = Field(default="通用", description="题目分类")


class Question(QuestionCreate):
    """完整面试题模型：继承请求体字段，追加服务端生成的字段"""

    id: int
    created_at: datetime


class QuestionUpdate(BaseModel):
    """更新面试题的请求体模型：所有字段可选，仅更新客户端提供的字段"""

    title: str | None = Field(default=None, min_length=5, max_length=200)
    category: str | None = Field(default=None)
