"""LLM 连通性冒烟测试：验证 API Key、网络与 SDK 封装是否可用"""

import sys
from pathlib import Path

# 把项目根目录加入模块搜索路径，使脚本能导入 app 包
# 直接运行 python scripts\smoke_llm.py 时，sys.path 只包含 scripts 目录
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.llm import chat


def main():
    """发送一条面试官角色的测试消息，打印模型回复"""
    reply = chat(
        messages=[
            {"role": "system", "content": "你是一名严格的技术面试官，回答简短"},
            {"role": "user", "content": "用一句话介绍你将如何面试候选人"},
        ],
        temperature=0.7,
    )
    print("模型回复：")
    print(reply)


if __name__ == "__main__":
    main()
