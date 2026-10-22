from fastapi import FastAPI

from app.routers import questions

# 创建 FastAPI 应用实例
# title/description/version 会展示在自动生成的文档页面上
app = FastAPI(
    title="AI Interviewer API",
    description="FastAPI 学习项目：面试题库 API",
    version="0.1.0",
)

# 挂载路由模块：include_router 把 router 里的所有端点注册进应用
# 后续新增资源时继续 include（如 users.router、interviews.router）
app.include_router(questions.router)


@app.get("/")
async def root():
    """根路由：返回欢迎信息，用于验证服务是否启动成功"""
    return {"message": "Hello FastAPI! 访问 /docs 查看交互式文档"}
