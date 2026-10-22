"""M11：节点级降级兜底——LLM 彻底不可用时，系统"优雅变差"而不是崩溃

降级分层（与 llm._call_with_backoff 的网络重试配合）：
- 第 1 层（llm.py）：网络抖动 → 指数退避重试，多数抖动在这一层自愈
- 第 2 层（本模块 + graph 节点）：LLM 重试耗尽仍失败 → 节点走降级路径：
  规划降级 = 兜底考点池轮换（维持"每题有方向"）
  出题降级   = 兜底题池轮换（维持"每轮有问题"）
  评估降级   = 中性默认分 + 降级标记（维持"每轮有结论"且不卡死流程）
  终评降级   = 基于 history 的统计式报告（维持"有交代"）

设计原则：
1. 降级产物必须能被下游正常消费（结构化类型不变，只是内容朴素）
2. 降级必须在轨迹里留痕（*_fallback 事件），"变差"要可审计
3. 评估降级的动作固定为 NEXT：FOLLOWUP 会原地打转，FINISH 则是
   "没评估就结束"——都不符合降级语义
"""

from app.interview.schemas import EvaluationResult, NextAction, PlannerDirective

# 兜底考点池：LLM 不可用时规划官退化为"按轮数轮换的固定考察方向"
FALLBACK_DIRECTIVES = [
    PlannerDirective(
        topic="Python 基础：数据结构与常用内置类型",
        guidance="出一道考察列表/字典/元组特性与选型的经典题，难度适中",
    ),
    PlannerDirective(
        topic="Python 进阶：装饰器与生成器",
        guidance="出一道考察装饰器原理或生成器惰性求值的经典题，难度适中",
    ),
    PlannerDirective(
        topic="后端工程：数据库与缓存基础",
        guidance="出一道考察索引/事务/缓存使用的经典题，难度适中",
    ),
]

# 兜底题目池：LLM 不可用时面试官退化为"按轮数轮换的经典通用题"
FALLBACK_QUESTIONS = [
    "请解释 Python 中列表和元组的区别，并说明在什么场景下你会选择元组？",
    "什么是装饰器？请手写一个记录函数执行耗时的装饰器，并解释其执行顺序。",
    "数据库索引为什么能加快查询？在什么情况下索引反而会变慢？",
]


def fallback_directive(round_count: int) -> PlannerDirective:
    """规划降级产物：按轮数从兜底考点池轮换取一个

    Args:
        round_count: 当前轮数（决定取哪条，保证多轮不重复）
    """
    return FALLBACK_DIRECTIVES[round_count % len(FALLBACK_DIRECTIVES)]


def fallback_question(round_count: int) -> str:
    """出题降级产物：按轮数从兜底题池轮换取一道

    Args:
        round_count: 当前轮数（决定取哪道，保证多轮不重复）
    """
    return FALLBACK_QUESTIONS[round_count % len(FALLBACK_QUESTIONS)]


def fallback_evaluation() -> EvaluationResult:
    """评估降级产物：中性默认分 + 显式降级标记

    给 5 分（中位数）而不是 0 分：降级不是候选人的错，不该惩罚候选人；
    comment 里写明降级原因，终评和 HR 审阅时能识别"这轮没有真实评估"。
    """
    return EvaluationResult(
        score=5,
        comment="（降级）评估服务暂时不可用，本轮按中性分处理，不代表候选人真实水平。",
        action=NextAction.NEXT,  # 不追问（会打转）、不结束（未评先终）——换下一题
    )


def fallback_summary(state: dict) -> str:
    """终评降级产物：不调 LLM，直接基于 history 拼一份统计式报告

    Args:
        state: 图的共享状态（读 position / candidate_name / history）
    """
    history = state.get("history", [])
    if not history:
        return "（降级）本场面试未能完成任何有效评估，无终评结论。"
    scores = [qa["score"] for qa in history]
    avg = sum(scores) / len(scores)
    lines = [
        f"（降级）LLM 终评服务不可用，以下为统计式摘要，仅供参考：",
        f"候选人：{state.get('candidate_name', '未知')}｜岗位：{state.get('position', '未知')}",
        f"共 {len(history)} 轮问答，平均分 {avg:.1f}（最高 {max(scores)} / 最低 {min(scores)}）。",
    ]
    for i, qa in enumerate(history, 1):
        lines.append(f"{i}. [{qa['score']} 分] {qa['question'][:40]}…")
    return "\n".join(lines)
