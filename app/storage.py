from datetime import datetime

from app.models import Question

# 内存模拟数据：服务重启即丢失，真实项目可替换为数据库（SQLAlchemy 等）
fake_questions: list[Question] = [
    Question(id=1, title="什么是 Python 的 GIL？", category="Python", created_at=datetime.now()),
    Question(id=2, title="HTTP 状态码 301 和 302 的区别？", category="网络", created_at=datetime.now()),
]
