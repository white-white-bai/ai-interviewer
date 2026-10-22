"""M4：会话记忆层——滑动窗口 + 滚动摘要压缩

问题（M2/M3 遗留）：_format_history 每轮把全部历史塞进出题提示词，
面试越长 token 越多——成本上升、响应变慢，长上下文还会稀释模型注意力。

策略（业界主流的"滑窗 + 滚动摘要"）：
- 最近 WINDOW_SIZE 轮问答保留原文（最近的内容对决定下一题最重要）
- 更早的轮次由 manage_memory 节点用一次小型 LLM 调用滚动合并进摘要
- 原始 history 一个字不删：最终总结仍读全量数据；
  压缩只发生在"喂给提示词"这一层——无损存储、有损注入
"""

from app import llm
from app.interview.prompts import MEMORY_COMPRESS_PROMPT

# 提示词中保留原文的最近问答轮数（滑动窗口大小）
WINDOW_SIZE = 3


def _format_items(items: list[dict], start: int) -> str:
    """把若干轮问答格式化为统一编号的文本（压缩提示词和记忆文本共用）

    Args:
        items: QAItem 字典列表
        start: 本批问答的起始编号（接在摘要轮次之后，保证编号连续）
    """
    lines = []
    for i, qa in enumerate(items, start):
        # 注意：不带 comment——评估意见只进摘要一次，不在每轮窗口里重复注入
        lines.append(
            f"{i}. 问：{qa['question']}\n"
            f"   答：{qa['answer']}（{qa['score']} 分）"
        )
    return "\n".join(lines)


def format_memory(state: dict) -> str:
    """把图的 state 组装成出题提示词用的"记忆文本"：早期摘要 + 窗口内原文

    与 M2 的 _format_history（全量拼接）对照：
    无论面试进行到多少轮，这里输出的长度只取决于滑窗大小和摘要长度，
    而不是随总轮数线性增长——这就是 M4 省 token 的核心。
    """
    history = state.get("history", [])
    upto = state.get("memory_upto", 0)  # 前 upto 轮已并入摘要
    parts = []
    if state.get("memory_summary"):
        parts.append(f"【早期问答摘要】\n{state['memory_summary']}")
    recent = history[upto:]  # 摘要之后的所有轮次（滑窗内原文）
    if recent:
        parts.append("【最近问答原文】\n" + _format_items(recent, start=upto + 1))
    if not parts:
        return "（暂无，这是第一个问题）"
    return "\n\n".join(parts)


def manage_memory(state: dict) -> dict:
    """记忆管理节点：把滑窗外的未压缩轮次滚动合并进摘要

    触发条件：history 里存在"滑窗外且尚未进摘要"的轮次
    （即 len(history) - WINDOW_SIZE > memory_upto）。
    没有新溢出时返回空 dict，对 state 不做任何修改——
    节点不必每次都干活，按需触发是记忆管理的常见形态。
    """
    history = state.get("history", [])
    upto = state.get("memory_upto", 0)
    overflow_end = len(history) - WINDOW_SIZE
    if overflow_end <= upto:
        return {}
    # 本次要压进摘要的轮次：[upto, overflow_end) 这一段
    new_items = history[upto:overflow_end]
    summary = llm.chat(
        messages=[{
            "role": "user",
            "content": MEMORY_COMPRESS_PROMPT.format(
                position=state["position"],
                existing_summary=state.get("memory_summary") or "（无）",
                new_text=_format_items(new_items, start=upto + 1),
            ),
        }],
        temperature=0.3,  # 低温度：摘要要忠实，不要发散
    )
    print(f"[记忆压缩] 第 {upto + 1}-{overflow_end} 轮问答已并入摘要")
    return {
        "memory_summary": summary.strip(),
        "memory_upto": overflow_end,  # 记录进度，下轮只压新增部分（滚动）
    }
