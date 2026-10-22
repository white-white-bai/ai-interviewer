"""面试题仓储层（Repository）：所有 SQL 集中在这里，路由层只见函数不见 SQL

仓储模式的要点：
- 上层（路由 / Agent 工具）只调用函数，不关心数据存在内存还是数据库
- M1 时这里是内存列表 fake_questions，M3 换成 SQLite，路由代码几乎零改动
"""

import sqlite3
from datetime import datetime

from app.db import db
from app.models import Question, QuestionCreate, QuestionUpdate


def _row_to_question(row: sqlite3.Row) -> Question:
    """把数据库行转成 Question 模型（created_at 从 ISO 字符串还原为 datetime）"""
    return Question(
        id=row["id"],
        title=row["title"],
        category=row["category"],
        created_at=datetime.fromisoformat(row["created_at"]),
    )


def get_question(question_id: int) -> Question | None:
    """按 ID 查题，不存在返回 None（抛不抛 404 由上层决定）"""
    with db() as conn:
        row = conn.execute(
            "SELECT * FROM questions WHERE id = ?", (question_id,)
        ).fetchone()
    return _row_to_question(row) if row else None


def search_questions(category: str = "", keyword: str = "") -> list[Question]:
    """按分类 / 关键词搜索题目（空字符串 = 不过滤该条件）

    WHERE 1=1 是拼条件的常用技巧：后续条件统一以 AND 开头。
    LIKE '%kw%' 做模糊匹配；参数一律用 ? 占位符传递，杜绝 SQL 注入。
    """
    sql = "SELECT * FROM questions WHERE 1=1"
    params: list = []
    if category:
        sql += " AND category = ?"
        params.append(category)
    if keyword:
        sql += " AND title LIKE ?"
        params.append(f"%{keyword}%")
    sql += " ORDER BY id"
    with db() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [_row_to_question(r) for r in rows]


def list_questions(page: int = 1, size: int = 10) -> tuple[list[Question], int]:
    """分页列出题目，返回 (当前页数据, 总条数)"""
    offset = (page - 1) * size
    with db() as conn:
        rows = conn.execute(
            "SELECT * FROM questions ORDER BY id LIMIT ? OFFSET ?",
            (size, offset),
        ).fetchall()
        total = conn.execute("SELECT COUNT(*) FROM questions").fetchone()[0]
    return [_row_to_question(r) for r in rows], total


def create_question(data: QuestionCreate) -> Question:
    """插入新题，返回带数据库自增 id 的完整模型"""
    now = datetime.now().isoformat()
    with db() as conn:
        cursor = conn.execute(
            "INSERT INTO questions (title, category, created_at) VALUES (?, ?, ?)",
            (data.title, data.category, now),
        )
        new_id = cursor.lastrowid  # 自增主键值
    return Question(id=new_id, created_at=datetime.fromisoformat(now), **data.model_dump())


def update_question(question_id: int, data: QuestionUpdate) -> Question | None:
    """部分更新：只 SET 客户端显式传入的字段；题目不存在返回 None

    SET 子句的列名来自 Pydantic 模型的字段名（非用户输入），
    值通过 ? 参数化传递，两处都不会引入 SQL 注入。
    """
    fields = data.model_dump(exclude_unset=True)
    if not fields:
        return get_question(question_id)  # 什么都没传，等价于查一次
    set_clause = ", ".join(f"{name} = ?" for name in fields)
    params = list(fields.values()) + [question_id]
    with db() as conn:
        conn.execute(f"UPDATE questions SET {set_clause} WHERE id = ?", params)
    return get_question(question_id)


def delete_question(question_id: int) -> bool:
    """删除题目，返回是否真的删除了行（rowcount 为受影响行数）"""
    with db() as conn:
        cursor = conn.execute("DELETE FROM questions WHERE id = ?", (question_id,))
    return cursor.rowcount > 0
