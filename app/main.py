from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.db import init_db
from app.routers import interview, interview_graph, questions

# 启动时初始化业务数据库：建表 + 空表写入种子数据（幂等，可重复执行）
init_db()

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
# LangGraph 版路由带固定段 /interview/graph/*，注册在含参数段的 interview 路由之前
app.include_router(interview_graph.router)
app.include_router(interview.router)


@app.get("/")
async def root():
    """根路由：返回欢迎信息，用于验证服务是否启动成功"""
    return {"message": "Hello FastAPI! 访问 /docs 查看交互式文档，访问 /static/index.html 进入面试前端"}


# 挂载静态文件目录（前端页面）：/static/index.html = 面试会话页，/static/questions.html = 题库管理页
# 同源托管：页面与 API 同域，fetch 调用无 CORS 问题。
# 目录用基于 __file__ 的绝对路径：相对路径会跟随"启动命令时的工作目录"，
# 换个目录启动 uvicorn 就会报 Directory 'static' does not exist 导致服务起不来
_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")
