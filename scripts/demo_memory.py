"""M4 演示：会话记忆——滑窗保留最近几轮原文 + 早期轮次滚动压缩为摘要

观察点：
1. history 始终全量保留（最终总结要用），但出题提示词注入的"记忆文本"
   = 摘要 + 最近 WINDOW_SIZE 轮，长度不再随轮数线性膨胀
2. 每行打印【全量注入】与【记忆注入】的字符数对比——差距从第 4 轮开始拉开
3. 控制台的 [记忆压缩] 行：manage_memory 节点触发了一次小型 LLM 压缩调用

候选人故意答得简略（低分 → followup），这样能凑够轮数让压缩触发。
真实调用约 15-20 次 LLM，DeepSeek 计费几毛钱以内。
"""

import sys
import uuid
from pathlib import Path

# 把项目根目录加入模块搜索路径（Windows 直接运行脚本时 sys.path 不含根目录）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langgraph.types import Command

from app.interview.graph import _format_history, interview_graph, make_config
from app.interview.memory import WINDOW_SIZE, format_memory
from app.llm import chat


def candidate_answer(question: str) -> str:
    """让 LLM 扮演一位回答简略含糊的候选人（演示需要多轮对话触发压缩）"""
    return chat(
        messages=[
            {
                "role": "system",
                "content": (
                    "你是一名 Python 后端候选人，回答技术问题时故意非常简略含糊，"
                    "每题只答一句话，只提概念不给细节（演示用途，为了让面试官多追问几轮）。"
                ),
            },
            {"role": "user", "content": question},
        ],
        temperature=0.7,
    )


def print_memory_report(values: dict, label: str) -> None:
    """打印一轮结束后的记忆状况：全量 vs 记忆注入的长度对比 + 摘要内容"""
    history = values.get("history", [])
    full_text = _format_history(history)
    memory_text = format_memory(values)
    upto = values.get("memory_upto", 0)
    print(
        f"[{label}] 已答 {len(history)} 轮 | 全量注入 {len(full_text)} 字 | "
        f"记忆注入 {len(memory_text)} 字 | 摘要覆盖前 {upto} 轮（滑窗={WINDOW_SIZE}）"
    )
    if values.get("memory_summary"):
        print(f"  当前摘要：{values['memory_summary']}")


def main():
    """跑一场最多 6 轮的模拟面试，逐轮观察记忆压缩效果"""
    thread_id = uuid.uuid4().hex[:8]  # 每次运行用新线程，避免读到旧存档
    print(f"线程ID（会话主键）: {thread_id}\n")

    # 第一次调用：图跑到 ask_candidate 的 interrupt() 暂停，出第一题
    interview_graph.invoke(
        {"candidate_name": "小明", "position": "Python 后端"},
        make_config(thread_id),
    )

    max_rounds = 6
    for round_no in range(1, max_rounds + 1):
        snap = interview_graph.get_state(make_config(thread_id))
        if not snap.next:  # next 为空说明已到 END
            break
        print(f"\n--- 第 {round_no} 轮 ---")
        question = snap.values["current_question"]
        print(f"面试官提问：{question}")
        answer = candidate_answer(question)
        print(f"候选人回答：{answer}\n")
        # 恢复执行：evaluate → manage_memory（可能触发压缩）→ 出下一题并暂停
        interview_graph.invoke(Command(resume=answer), make_config(thread_id))
        print_memory_report(
            interview_graph.get_state(make_config(thread_id)).values or {},
            f"第 {round_no} 轮后",
        )

    final = interview_graph.get_state(make_config(thread_id)).values or {}
    if final.get("summary"):
        print(f"\n===== 面试结束（共 {len(final.get('history', []))} 轮问答）=====")
        print(f"面试官总评（基于全量 history 生成）：{final.get('summary')}")
    else:
        print(
            f"\n（达到演示轮数上限；检查点仍在，之后可对同一 thread_id 继续 resume）"
        )


if __name__ == "__main__":
    main()
