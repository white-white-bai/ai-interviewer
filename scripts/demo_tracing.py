"""M10 演示：轨迹评测——把"验收 Agent 行为"从肉眼读日志变成程序断言

两步：
Part 1 造数据：跑一场真实面试（2 轮答题 + 注入 finish + 批准收尾，
  复用 M8 的确定性推进手法），全程事件自动落 traces/{thread_id}.jsonl
Part 2 评测：读回轨迹文件，对行为做断言——
  流程完整性（plan/evaluate/sandbox/approval/summary 数量关系）、
  值域合法（分数 0-10、动作枚举）、重试预算（self_correction ≤ 上限）、
  审计规则（每次 finish 都有审批记录、终评只在批准后生成）、
  时间戳单调（JSONL 顺序可信）。

教学点：可观测性是 Agent 工程化的前提——
M9 之前行为对不对靠 print 人肉判断，M10 之后靠轨迹断言自动验收。

全程约 8-10 分钟（约 15 次 LLM 调用），DeepSeek 计费几毛钱。
"""

import sys
import uuid
from pathlib import Path

# 把项目根目录加入模块搜索路径（Windows 直接运行脚本时 sys.path 不含根目录）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langgraph.types import Command

from app.interview import tracing
from app.interview.graph import interview_graph, make_config
from app.llm import chat

WARMUP_ROUNDS = 2  # 真实答题轮数：造轨迹用，想快可改成 1


def candidate_answer(question: str) -> str:
    """让 LLM 扮演优秀候选人：回答专业深入（评测质量好坏不影响演示结构）"""
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
    """人工注入 finish 决策：update_state 接管评估官，保证演示确定性（M8 手法）"""
    interview_graph.update_state(
        make_config(thread_id),
        {"action": "finish"},
        as_node="manage_memory",
    )
    interview_graph.invoke(None, make_config(thread_id))


def run_interview(thread_id: str) -> None:
    """跑一场完整面试造轨迹：2 轮真实答题 → 注入 finish → 批准 → 终评"""
    interview_graph.invoke(
        {"candidate_name": "小测", "position": "Python 后端"},
        make_config(thread_id),
    )
    for round_no in range(1, WARMUP_ROUNDS + 1):
        snap = interview_graph.get_state(make_config(thread_id))
        if not snap.next or snap.next[0] == "approval":
            break  # 评估官自然收尾了就直接用
        question = snap.values["current_question"]
        print(f"\n--- 第 {round_no} 轮 ---")
        print(f"面试官提问：{question[:100]}…")
        answer = candidate_answer(question)
        print(f"候选人回答（摘要）：{answer[:60]}…")
        interview_graph.invoke(Command(resume=answer), make_config(thread_id))
        values = interview_graph.get_state(make_config(thread_id)).values or {}
        print(f"[第 {round_no} 轮评估] {values.get('last_score')} 分 | 动作={values.get('action')}")
    print(f"[推进] {WARMUP_ROUNDS} 轮完成，注入 finish（update_state 接管）")
    force_finish(thread_id)
    print("[人工] 批准 → 生成终评")
    interview_graph.invoke(Command(resume=True), make_config(thread_id))


def eval_trace(thread_id: str) -> int:
    """对轨迹做程序化断言，打印每条 PASS/FAIL，返回失败条数

    断言分四类：流程完整性 / 值域合法 / 预算与审计 / 数据可信。
    这就是"评测"的具象——以后每次改提示词/模型/图结构，
    重跑一遍就知道行为有没有退化（回归测试的雏形）。
    """
    events = tracing.load_trace(thread_id)
    print(f"\n===== Part 2：轨迹评测（共 {len(events)} 条事件）=====")
    failures = 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        """单条断言：打印结果并累计失败数"""
        nonlocal failures
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {name}" + (f" —— {detail}" if detail else ""))
        if not ok:
            failures += 1

    plans = tracing.filter_events(events, "plan")
    evaluates = tracing.filter_events(events, "evaluate")
    sandboxes = tracing.filter_events(events, "sandbox")
    approvals_req = tracing.filter_events(events, "approval_requested")
    approvals = tracing.filter_events(events, "approval")
    corrections = tracing.filter_events(events, "self_correction")
    summaries = tracing.filter_events(events, "summary")

    # --- 流程完整性：图拓扑在轨迹上的数量关系 ---
    check("每轮都有规划（plan ≥ 1）", len(plans) >= 1, f"plan={len(plans)}")
    # 注意是 ≥ 不是 ==：演示用 update_state 注入 finish 会中断"已规划未评估"的
    # 下一题，plan 比 evaluate 多 1 是预期行为（首次验证时实测抓出该差异）
    check("每道被评估的题都先有规划（plan ≥ evaluate）",
          len(plans) >= len(evaluates), f"plan={len(plans)} evaluate={len(evaluates)}")
    check("每轮都过沙箱（sandbox 数 = evaluate 数）",
          len(sandboxes) == len(evaluates), f"sandbox={len(sandboxes)}")
    check("终评恰好一次", len(summaries) == 1, f"summary={len(summaries)}")

    # --- 值域合法：LLM 结构化输出的约束真的被满足 ---
    check("所有评分在 0-10",
          all(0 <= e.get("score", -1) <= 10 for e in evaluates))
    check("所有动作是合法枚举",
          all(e.get("action") in {"followup", "next", "finish"} for e in evaluates))

    # --- 预算与审计：M9 重试预算 + M8 不可逆动作审计 ---
    check("自我纠错都在预算内（单次 ≤ 2 次）",
          all(e.get("attempt", 0) <= 2 for e in corrections),
          f"self_correction={len(corrections)} 次")
    # 注意 approval_requested 会因 interrupt 重放语义记录多次（暂停前一次 +
    # resume 后节点重放再一次），这是 LangGraph 的正常行为；审计要求是
    # "有请求、有决定、决定不比请求多"
    check("审批链完整（请求 ≥ 决定 ≥ 1）",
          len(approvals_req) >= len(approvals) >= 1,
          f"requested={len(approvals_req)} decided={len(approvals)}")
    check("终评只在批准后生成（存在 approved=True）",
          len(summaries) == 0 or any(e.get("approved") for e in approvals))

    # --- 数据可信：JSONL 追加顺序即时间顺序 ---
    ts_list = [e.get("ts", "") for e in events]
    check("时间戳单调不减（追加写可信）", ts_list == sorted(ts_list))

    # 展示前几条事件，直观感受轨迹长什么样
    print("\n轨迹样例（前 5 条）：")
    for e in events[:5]:
        keys = {k: v for k, v in e.items() if k not in {"ts", "thread_id"}}
        print(f"  {e['ts']} {e['event']} {str(keys)[:100]}")
    return failures


def main() -> None:
    """跑面试造轨迹 → 断言评测 → 汇总结果

    支持 --eval-only <thread_id>：跳过跑面试，直接评测已有轨迹
    （修断言后复核用，不调 LLM、零花费、秒出）。
    """
    if len(sys.argv) == 3 and sys.argv[1] == "--eval-only":
        failures = eval_trace(sys.argv[2])
    else:
        thread_id = uuid.uuid4().hex[:8]  # 每次运行用新线程：轨迹文件独立、可复现
        print(f"===== Part 1：跑一场面试造轨迹（thread_id={thread_id}）=====")
        print(f"轨迹文件：traces/{thread_id}.jsonl")
        run_interview(thread_id)
        failures = eval_trace(thread_id)
    if failures == 0:
        print("\n===== M10 评测通过：全部断言 PASS =====")
    else:
        print(f"\n===== M10 评测发现 {failures} 条断言失败（行为回归，需要排查）=====")
        sys.exit(1)  # 非零退出码：方便以后接 CI


if __name__ == "__main__":
    main()
