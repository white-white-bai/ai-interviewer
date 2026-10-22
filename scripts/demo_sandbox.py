"""M6 演示：代码沙箱——隔离执行 + 超时强杀 + 评估官基于真实运行结果打分

Part 1：沙箱单元演示（零 LLM 调用）——三种典型结果：
  正常输出 / 运行报错 / 死循环超时强杀
Part 2：完整面试轮——候选人每答必附 ```python 代码块，
  观察图中 [沙箱] 节点真实执行代码，评估官引用运行结果打分

图的执行路径变化：ask_candidate → run_code（新增）→ evaluate → ...
真实调用约 8-10 次 LLM，DeepSeek 计费几分钱。
"""

import sys
import uuid
from pathlib import Path

# 把项目根目录加入模块搜索路径（Windows 直接运行脚本时 sys.path 不含根目录）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langgraph.types import Command

from app.interview.graph import interview_graph, make_config
from app.llm import chat
from app.sandbox import run_python

# Part 1 的三段测试代码：覆盖沙箱的三种典型结局
SANDBOX_CASES = [
    ("正常代码", "print(sum(range(10)))"),
    ("运行报错", "print(1 / 0)"),
    ("死循环", "while True:\n    pass"),
]


def demo_sandbox_units() -> None:
    """Part 1：直接调沙箱跑三段代码，观察隔离与超时机制（不经过 LLM）"""
    print("===== Part 1：沙箱单元演示 =====")
    for name, code in SANDBOX_CASES:
        result = run_python(code, timeout_seconds=3.0)
        print(f"\n[{name}] ok={result['ok']}, timed_out={result['timed_out']}")
        print(f"  stdout: {result['stdout'].strip()[:80]!r}")
        print(f"  stderr: {result['stderr'].strip()[:80]!r}")


def candidate_answer(question: str) -> str:
    """让 LLM 扮演候选人：无论被问什么都附代码块（保证触发沙箱节点）"""
    return chat(
        messages=[
            {
                "role": "system",
                "content": (
                    "你是一名 Python 后端候选人。无论面试官问什么，回答末尾都附一个"
                    " ```python 代码块来演示你的观点（代码可以有 bug，照写不误）。"
                ),
            },
            {"role": "user", "content": question},
        ],
        temperature=0.7,
    )


def main() -> None:
    """先跑沙箱单元演示，再跑两轮带代码块的模拟面试"""
    demo_sandbox_units()

    print("\n===== Part 2：带沙箱的模拟面试 =====")
    thread_id = uuid.uuid4().hex[:8]  # 每次运行用新线程，避免读到旧存档
    print(f"线程ID（会话主键）: {thread_id}")

    # 第一次调用：规划 → 出题 → 暂停在 interrupt
    interview_graph.invoke(
        {"candidate_name": "小明", "position": "Python 后端"},
        make_config(thread_id),
    )

    for round_no in range(1, 3):  # 演示 2 轮即可观察机制
        snap = interview_graph.get_state(make_config(thread_id))
        if not snap.next:  # next 为空说明已到 END
            break
        print(f"\n--- 第 {round_no} 轮 ---")
        question = snap.values["current_question"]
        print(f"面试官提问：{question}")
        answer = candidate_answer(question)
        print(f"候选人回答：{answer}\n")
        # 恢复执行：run_code（可能触发沙箱）→ 评估 → 记忆 → 规划 → 出题 → 暂停
        interview_graph.invoke(Command(resume=answer), make_config(thread_id))
        values = interview_graph.get_state(make_config(thread_id)).values or {}
        print(
            f"[第 {round_no} 轮评估] {values.get('last_score')} 分 | "
            f"动作={values.get('action')}"
        )
        print(f"  评估官意见：{values.get('last_comment')}")

    final = interview_graph.get_state(make_config(thread_id)).values or {}
    if not interview_graph.get_state(make_config(thread_id)).next:
        print(f"\n===== 面试结束 =====\n面试官总评：{final.get('summary')}")
    else:
        print("\n（达到演示轮数上限；检查点仍在，可继续 resume）")


if __name__ == "__main__":
    main()
