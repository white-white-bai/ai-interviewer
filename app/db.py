"""SQLite 数据库连接与初始化（M3：业务数据落库）

为什么用标准库 sqlite3 而不是 SQLAlchemy？
- 学习重点在 Agent 工程而非 ORM，原生 SQL 更透明
- storage.py 仓储层已把 SQL 隔离，未来换 PostgreSQL/ORM 只改这一个文件
"""

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

# 数据库文件：项目根目录 interviewer.db（已被 .gitignore 的 *.db 规则忽略）
DB_PATH = Path(__file__).resolve().parent.parent / "interviewer.db"

# 建表语句：IF NOT EXISTS 保证重复执行安全（幂等）
_SCHEMA = """
CREATE TABLE IF NOT EXISTS questions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    title      TEXT NOT NULL,
    category   TEXT NOT NULL DEFAULT '通用',
    created_at TEXT NOT NULL
);
"""

# 种子数据：仅当表为空时写入，避免每次启动重复插入
_SEED_QUESTIONS = [
    ("什么是 Python 的 GIL？它对多线程程序有什么影响？", "Python"),
    ("HTTP 状态码 301 和 302 的区别是什么？", "网络"),
    ("FastAPI 的依赖注入 Depends 解决了什么问题？", "FastAPI"),
    ("数据库索引的原理是什么？什么情况下索引会失效？", "数据库"),
    ("git merge 和 git rebase 的区别是什么？", "工程工具"),
]


@contextmanager
def db():
    """数据库连接上下文：自动提交 / 回滚 / 关闭

    sqlite3 的连接默认不允许跨线程使用，而 FastAPI 会用线程池并发
    调用同步路由函数；"每次操作开新连接"是单机小项目最简单可靠的
    并发方案（SQLite 开连接成本极低）。
    用法：with db() as conn: conn.execute(...)
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # 查询结果可用列名访问：row["title"]
    try:
        yield conn
        conn.commit()  # 正常结束：提交事务
    except Exception:
        conn.rollback()  # 出错：回滚，保证不写脏数据
        raise
    finally:
        conn.close()


def init_db() -> None:
    """初始化数据库：建表 + 空表时写入种子数据（在 main.py 启动时调用一次）"""
    with db() as conn:
        conn.executescript(_SCHEMA)
        # COUNT(*) 返回单行单列，取第一列判断表是否为空
        is_empty = conn.execute("SELECT COUNT(*) FROM questions").fetchone()[0] == 0
        if is_empty:
            conn.executemany(
                "INSERT INTO questions (title, category, created_at) VALUES (?, ?, ?)",
                [
                    (title, category, datetime.now().isoformat())
                    for title, category in _SEED_QUESTIONS
                ],
            )
