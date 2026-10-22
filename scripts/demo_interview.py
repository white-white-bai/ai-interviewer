"""面试流程端到端演示：LLM 同时扮演面试官与候选人，观察状态机流转

真实调用约 6-8 次 LLM，花费极低（DeepSeek 计费约几分钱）。
"""

import sys
from pathlib import Path

# 把项目根目录加入模块搜索路径（与 smoke_llm.py 相同的原因）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.interview import service
from app.interview.schemas import InterviewSession
from app.interview.state_machine import InterviewState
from app.llm import chat

# 演示最大问答轮数，防止 LLM 一直出题不结束
MAX_ROUNDS = 3


def candidate_answer(question: str) -> str:
    """让 LLM 扮演候选人回答面试题（演示用，真实场景来自用户输入）"""
    return chat(
        messages=[
            {
                "role": "system",
                "content": "你是一名 Python 后端候选人，回答技术问题简洁真实。",
            },
            {"role": "user", "content": question},
        ],
        temperature=0.7,
    )


def print_state(session: InterviewSession, label: str) -> None:
    """打印会话状态快照，便于观察状态机流转"""
    print(
        f"[{label}] state={session.state.value}, "
        f"已答题={len(session.history)}, 追问={session.followup_count}"
    )
    if session.current_question:
        print(f"  当前题目：{session.current_question}")


def main():
    """跑一场最多 3 轮问答的模拟面试"""
    # 1. 创建会话并生成第一题：CREATED → WAITING_ANSWER
    session = service.start_interview("小明", "Python 后端")
    print(f"会话ID: {session.id}")
    print_state(session, "开始")

    # 2. 循环问答：每次提交回答后观察状态如何流转
    for round_no in range(1, MAX_ROUNDS + 1):
        answer = candidate_answer(session.current_question)
        print(f"\n--- 第 {round_no} 轮 ---")
        print(f"候选人回答：{answer}\n")

        session = service.submit_answer(session.id, answer)
        qa = session.history[-1]
        print(f"AI 评估：{qa.score} 分 - {qa.comment}")
        print_state(session, "评估后")

        if session.state is InterviewState.FINISHED:
            break

    print(f"\n===== 面试结束：共 {len(session.history)} 轮问答 =====")


if __name__ == "__main__":
    main()
