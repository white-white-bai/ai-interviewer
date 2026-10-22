"""最小 MCP 服务器（M3-c）：把 search_question_bank 以标准协议暴露出去

MCP（Model Context Protocol）把工具搬进独立进程：
- 工具用 @mcp.tool() 装饰器注册，函数名 + docstring 就是工具的
  名称与描述（协议会自动转成客户端能读懂的 schema）
- 本文件用 stdio 传输：客户端会以子进程方式启动本脚本，
  通过标准输入/输出通信（另一种传输是 Streamable HTTP）

运行方式：不需要手动启动，客户端脚本会自动把它作为子进程拉起。
任何 MCP 客户端（Claude Desktop、其他 Agent）都能发现并调用这里的工具。
"""

import json
import sys
from pathlib import Path

# 确保能 import app 包（脚本可能从任意目录被启动）
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mcp.server.fastmcp import FastMCP

from app import storage

# 创建 MCP 服务器实例，name 会展示给客户端
mcp = FastMCP("interviewer-tools")


@mcp.tool()
def search_question_bank(keyword: str = "", category: str = "") -> str:
    """查询面试题库，按分类和关键词检索已有题目

    Args:
        keyword: 题目关键词，如 GIL、索引、事务
        category: 题目分类，如 Python、数据库、网络
    """
    # 复用 M3-a 的仓储层：MCP 服务器只是"换个协议暴露同一份能力"
    results = storage.search_questions(category=category, keyword=keyword)
    payload = [{"id": q.id, "title": q.title, "category": q.category} for q in results]
    return json.dumps({"count": len(payload), "questions": payload}, ensure_ascii=False)


if __name__ == "__main__":
    # stdio 模式：阻塞在标准输入上等待客户端请求
    mcp.run()
