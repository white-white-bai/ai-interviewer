"""M5 演示：多 Agent 协作——规划官 → 面试官 → 评估官 各司其职

与 M4 的 demo_memory.py 相比，每轮多了一环（看控制台角色日志）：
[规划官] 下一考点：xxx          ← plan_topic 节点，决定追问还是换题
[面试官·工具调用] ...           ← generate_question 节点，按指令定向出题
（评估结果在每轮末尾打印）      ← evaluate 节点，独立人设打分

观察点：
1. 规划官是否真的"指挥"了面试官——题目是否命中规划的考点
2. 低分轮次规划官是否选择"继续深入"（追问），而非盲目换题
3. Agent 之间没有直接对话：协作全靠 state["directive"] 传递

真实调用约 12-16 次 LLM，DeepSeek 计费几毛钱以内。
"""

import sys
import uuid
from pathlib import Path

# 把项目根目录加入模块搜索路径（Windows 直接运行脚本时 sys.path 不含根目录）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langgraph.types import Command

from app.interview.graph import interview_graph, make_config
from app.llm import chat


def candidate_answer(question: str) -> str:
    """让 LLM 扮演候选人回答面试题（演示用，真实场景来自用户输入）"""
    return chat(
        messages=[
            {
                "role": "system",
                "content": "你是一名 Python 后端候选人，回答技术问题简洁真实，遇到不熟的领域就坦白说不会。",
            },
            {"role": "user", "content": question},
        ],
        temperature=0.7,
    )


def print_round_result(values: dict, round_no: int) -> None:
    """打印一轮结束后的评估结果，方便观察"规划 ↔ 评估"的闭环"""
    print(
        f"[第 {round_no} 轮评估] {values.get('last_score')} 分 | "
        f"动作={values.get('action')} | 累计 {len(values.get('history', []))} 轮"
    )
    print(f"  评估官意见：{values.get('last_comment')}")


def main():
    """跑一场最多 4 轮的多 Agent 模拟面试"""
    thread_id = uuid.uuid4().hex[:8]  # 每次运行用新线程，避免读到旧存档
    print(f"线程ID（会话主键）: {thread_id}\n")

    # 第一次调用：plan_topic → generate_question → 暂停在 interrupt
    interview_graph.invoke(
        {"candidate_name": "小明", "position": "Python 后端"},
        make_config(thread_id),
    )

    max_rounds = 4
    for round_no in range(1, max_rounds + 1):
        snap = interview_graph.get_state(make_config(thread_id))
        if not snap.next:  # next 为空说明已到 END
            break
        print(f"\n--- 第 {round_no} 轮 ---")
        question = snap.values["current_question"]
        print(f"面试官提问：{question}")
        answer = candidate_answer(question)
        print(f"候选人回答：{answer}\n")
        # 恢复执行：评估 → 记忆管理 → 条件边 → 规划 → 出题 → 再次暂停
        interview_graph.invoke(Command(resume=answer), make_config(thread_id))
        values = interview_graph.get_state(make_config(thread_id)).values or {}
        print_round_result(values, round_no)

    final = interview_graph.get_state(make_config(thread_id)).values or {}
    if final.get("summary"):
        print(f"\n===== 面试结束（共 {len(final.get('history', []))} 轮问答）=====")
        print(f"面试官总评：{final.get('summary')}")
    else:
        print("\n（达到演示轮数上限；检查点仍在，之后可对同一 thread_id 继续 resume）")


if __name__ == "__main__":
    main()
