"""LangGraph 版面试端到端演示：观察 interrupt 暂停 / Command 恢复 / 检查点读档

真实调用约 6-10 次 LLM，花费极低（DeepSeek 计费约几分钱）。
运行后项目根目录会出现 checkpoints.sqlite——那就是 LangGraph 的存档文件。
"""

import sys
import uuid
from pathlib import Path

# 把项目根目录加入模块搜索路径（Windows 直接运行脚本时 sys.path 不含根目录）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langgraph.types import Command

from app.interview.graph import interview_graph, make_config
from app.llm import chat

# 演示最大问答轮数，防止 LLM 一直追问不结束
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


def print_snapshot(thread_id: str, label: str) -> bool:
    """从检查点读档并打印快照；返回图是否已跑到 END

    snap.next 非空 = 图暂停在 interrupt 处等待输入；为空 = 已到 END。
    """
    snap = interview_graph.get_state(make_config(thread_id))
    values = snap.values or {}
    status = "等待回答" if snap.next else "已结束"
    print(
        f"[{label}] next={list(snap.next) or 'END'}, 状态={status}, "
        f"已答题={len(values.get('history', []))}, 出题数={values.get('round_count', 0)}"
    )
    if values.get("current_question"):
        print(f"  当前题目：{values['current_question']}")
    return not snap.next


def main():
    """跑一场最多 3 轮问答的 LangGraph 模拟面试"""
    thread_id = uuid.uuid4().hex[:8]  # 每次运行用新线程，避免读到旧存档
    print(f"线程ID（会话主键）: {thread_id}\n")

    # 第一次调用：图从 START 跑到 ask_candidate 的 interrupt() 处自动暂停，
    # 此刻状态已写入 checkpoints.sqlite——这就是"存档"
    interview_graph.invoke(
        {"candidate_name": "小明", "position": "Python 后端"},
        make_config(thread_id),
    )
    finished = print_snapshot(thread_id, "第一次调用（图已暂停）")

    round_no = 0
    while not finished and round_no < MAX_ROUNDS:
        round_no += 1
        print(f"\n--- 第 {round_no} 轮 ---")
        # 读档拿到当前题目，让 LLM 扮演候选人作答
        snap = interview_graph.get_state(make_config(thread_id))
        answer = candidate_answer(snap.values["current_question"])
        print(f"候选人回答：{answer}\n")
        # Command(resume=...) = "读档续玩"：interrupt() 返回该答案，
        # 图继续 evaluate → 条件边 → 出下一题并再次暂停，或 summarize 后到 END
        interview_graph.invoke(Command(resume=answer), make_config(thread_id))
        finished = print_snapshot(thread_id, "恢复执行后")

    final = interview_graph.get_state(make_config(thread_id)).values or {}
    if finished:
        print(f"\n===== 面试结束（共 {len(final.get('history', []))} 轮问答）=====")
        print(f"面试官总评：{final.get('summary')}")
    else:
        print(
            "\n（达到演示轮数上限；检查点仍在，之后可对同一 thread_id 继续 resume 接着面）"
        )


if __name__ == "__main__":
    main()
