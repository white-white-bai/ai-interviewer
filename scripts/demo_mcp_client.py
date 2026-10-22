"""MCP 客户端演示：连接 MCP 服务器，发现工具并调用（M3-c）

完整演示 MCP 的三步握手：
1. stdio_client 把 mcp_server.py 作为子进程拉起，建立双向管道
2. session.initialize() 完成协议握手，list_tools() 发现服务器有哪些工具
3. call_tool() 跨进程调用工具，拿回结果

对比 function calling：这里没有 LLM 参与——MCP 只负责"发现与调用"，
模型决定"何时调用"仍然是 function calling 的事。两者是互补关系。
"""

import asyncio
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# 服务器脚本路径：项目根目录的 mcp_server.py
SERVER_SCRIPT = Path(__file__).resolve().parent.parent / "mcp_server.py"


async def main():
    """连接 MCP 服务器 → 列出工具 → 调用一次工具查询"""
    # 用当前 Python 解释器启动服务器子进程（保证用的是同一个 venv）
    params = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER_SCRIPT)],
    )
    # stdio_client：通过子进程的 stdin/stdout 与服务器通信
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            # 协议握手：交换版本与能力声明
            await session.initialize()

            # 发现工具：客户端无需硬编码，服务器"有什么"由它自己声明
            tools = await session.list_tools()
            print("=== 服务器暴露的工具 ===")
            for tool in tools.tools:
                print(f"- {tool.name}: {tool.description}")

            # 跨进程调用工具：与在本地调函数一样自然
            print("\n=== 调用 search_question_bank(keyword='索引') ===")
            result = await session.call_tool(
                "search_question_bank", {"keyword": "索引"}
            )
            print(result.content[0].text)


if __name__ == "__main__":
    asyncio.run(main())
