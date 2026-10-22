"""M8 演示：人工审批——不可逆动作（结束面试 + 生成终评）前的人类把关

Part 1（批准路径）：真实答题 2 轮 → 人工注入 finish → 图暂停在 approval
  → 人工批准（resume=True）→ summarize 生成终评 → END
Part 2（驳回路径）：走到 approval → 人工驳回（resume=False）
  → 图回 plan_topic 继续面试 → 再次注入 finish → 这次批准 → 正常收尾

为什么需要"注入 finish"：评估官何时决定收尾有随机性（第一次验证时
两段各 6 轮都没触发 finish），演示改用 update_state 人工接管决策，
保证审批断点一定出现——这本身也是 LangGraph 的官方人工干预机制。

教学点：interrupt 不只能"等输入"，还能做"高风险决策的人工把关"——
Agent 可以自主建议，但不可逆动作的执行权在人类手里。
真实调用约 20-25 次 LLM，DeepSeek 计费几毛钱。
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
    """让 LLM 扮演优秀候选人：回答专业深入（评估质量好坏不影响演示结构）"""
    return chat(
        messages=[
            {
                "role": "system",
                "content": (
                    "你是一名非常优秀的 Python 后端候选人。回答要专业、结构清晰、"
                    "覆盖全面并给出关键细节，力求表现优异，让面试官觉得无需再追问。"
                ),
            },
            {"role": "user", "content": question},
        ],
        temperature=0.7,
    )


def force_finish(thread_id: str) -> None:
    """人工注入 finish 决策：把图推进到审批断点（保证演示确定性）

    update_state 的 as_node="manage_memory" 表示"这次状态修改伪装成
    manage_memory 节点的输出"，随后 invoke(None) 让图从该节点的出边
    重新路由：route_after_evaluate 读到 action="finish" → 走向 approval →
    interrupt() 暂停。这是 LangGraph 官方的"人工接管 Agent 决策"机制。
    """
    interview_graph.update_state(
        make_config(thread_id),
        {"action": "finish"},
        as_node="manage_memory",
    )
    # None 输入 = 不提供新输入，从上次状态继续执行（一路跑到下一个 interrupt）
    interview_graph.invoke(None, make_config(thread_id))


def drive_to_approval(thread_id: str, warmup_rounds: int = 2) -> str:
    """先真实答题 warmup_rounds 轮（演示完整流程），再注入 finish 推到审批点

    每轮：读档 → 取当前问题 → 生成回答 → resume 恢复执行
    （评估 → 记忆 → 条件边 → 出下一题并再次暂停）。
    若评估官自己提前决定收尾（action=finish），也直接算到达审批点。

    Returns:
        "approval"（等待人工审批）或 "finished"（意外提前到 END）
    """
    for round_no in range(1, warmup_rounds + 1):
        snap = interview_graph.get_state(make_config(thread_id))
        if not snap.next:
            return "finished"
        if snap.next[0] == "approval":  # 评估官自然触发了结束决策
            print(f"[推进] 评估官决定结束面试，图暂停在审批点（第 {round_no - 1} 轮评估后）")
            return "approval"
        question = snap.values["current_question"]
        print(f"\n--- 第 {round_no} 轮 ---")
        print(f"面试官提问：{question}")
        answer = candidate_answer(question)
        print(f"候选人回答（摘要）：{answer[:100]}…")
        interview_graph.invoke(Command(resume=answer), make_config(thread_id))
        values = interview_graph.get_state(make_config(thread_id)).values or {}
        print(
            f"[第 {round_no} 轮评估] {values.get('last_score')} 分 | "
            f"动作={values.get('action')}"
        )
    # 人工接管：注入 finish，把图推到审批断点
    print(f"[推进] {warmup_rounds} 轮演示完成，人工注入 finish 决策（update_state 接管评估官）")
    force_finish(thread_id)
    snap = interview_graph.get_state(make_config(thread_id))
    if snap.next and snap.next[0] == "approval":
        print("[推进] 图暂停在审批点")
        return "approval"
    return "finished"


def part1_approve() -> None:
    """Part 1：批准路径——人工批准 finish 决策 → 生成终评"""
    print("===== Part 1：批准路径 =====")
    thread_id = uuid.uuid4().hex[:8]  # 每次运行用新线程，避免读到旧存档
    print(f"线程ID（会话主键）: {thread_id}")
    interview_graph.invoke(
        {"candidate_name": "小明", "position": "Python 后端"},
        make_config(thread_id),
    )
    if drive_to_approval(thread_id) != "approval":
        print("（未能到达审批断点，演示提前结束）")
        return
    values = interview_graph.get_state(make_config(thread_id)).values or {}
    print(f"审批上下文：已完成 {values.get('round_count')} 轮，最近得分 {values.get('last_score')} 分")
    print("[人工] 决定：批准 → 生成终评")
    interview_graph.invoke(Command(resume=True), make_config(thread_id))
    final = interview_graph.get_state(make_config(thread_id)).values or {}
    print(f"\n===== 面试结束 =====\n终评：{final.get('summary')}")


def part2_reject() -> None:
    """Part 2：驳回路径——先驳回继续面，再走到审批点时批准收尾"""
    print("\n\n===== Part 2：驳回路径 =====")
    thread_id = uuid.uuid4().hex[:8]
    print(f"线程ID（会话主键）: {thread_id}")
    interview_graph.invoke(
        {"candidate_name": "小红", "position": "Python 后端"},
        make_config(thread_id),
    )
    if drive_to_approval(thread_id) != "approval":
        print("（未能到达审批断点，演示提前结束）")
        return
    print("[人工] 决定：驳回 → 面试继续（图回规划官，检查点保证现场不丢）")
    interview_graph.invoke(Command(resume=False), make_config(thread_id))
    snap = interview_graph.get_state(make_config(thread_id))
    print(f"[驳回后] 图重新流转，当前停在：{list(snap.next)}（等待回答）")
    # 再来一轮真实答题 + 注入 finish，这次批准收尾
    if drive_to_approval(thread_id) == "approval":
        print("[人工] 决定：这次批准 → 生成终评")
        interview_graph.invoke(Command(resume=True), make_config(thread_id))
        final = interview_graph.get_state(make_config(thread_id)).values or {}
        print(f"\n===== 面试结束 =====\n终评：{final.get('summary')}")
    else:
        print("（未能再次到达审批断点；检查点仍在，可继续 resume）")


def main() -> None:
    """依次演示批准与驳回两条路径"""
    part1_approve()
    part2_reject()
    print("\n===== M8 演示结束 =====")


if __name__ == "__main__":
    main()
