"""M10：轨迹层——把 Agent 每一步决策结构化记录成"可评测的证据"

M2-M9 各节点里的 print 是教学期的临时观察手段：给人看，但程序没法验收。
M10 把关键事件（规划/工具调用/沙箱/评估/纠错/审批/总结）落成 JSONL：

- 一场面试一个文件（traces/{thread_id}.jsonl），一行一条事件
- print 给人看，traces 给程序看——评测脚本直接对轨迹做断言（见
  scripts/demo_tracing.py），"验收"从肉眼读日志变成程序化检查

为什么选 JSONL 自建而不是 LangSmith/Langfuse：
- 零新依赖、格式自己定义、断言随便写——教学场景足够
- 托管方案的取舍：可视化强、检索方便，但引入账号/网络/费用依赖，
  且轨迹一旦出本地环境就多了一层数据合规考量（生产再权衡）

对比 M10 之前的临时 print：print 是"人肉评测"，轨迹是"机器评测"。
"""

import json
import threading
from datetime import datetime
from pathlib import Path

# 轨迹目录：项目根/traces/，一场面试一个 JSONL 文件
TRACE_DIR = Path(__file__).resolve().parent.parent.parent / "traces"

# FastAPI 多线程处理请求，多个面试会话可能同时写盘，追加写需要互斥
_lock = threading.Lock()


def log_event(thread_id: str, event: str, **fields) -> dict:
    """记录一条轨迹事件：追加写入 traces/{thread_id}.jsonl

    Args:
        thread_id: 面试线程 ID（会话主键，决定写到哪个文件）
        event: 事件类型，如 plan / tool_call / tool_result / sandbox /
               evaluate / self_correction / approval / summary
        **fields: 事件细节（任意可 JSON 序列化的字段，由调用方决定）

    Returns:
        完整事件 dict（含时间戳），方便调用方复用
    """
    record = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "thread_id": thread_id,
        "event": event,
        **fields,
    }
    TRACE_DIR.mkdir(exist_ok=True)
    with _lock:
        with open(TRACE_DIR / f"{thread_id}.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def load_trace(thread_id: str) -> list[dict]:
    """读取一场面试的完整轨迹（评测脚本用）

    Args:
        thread_id: 面试线程 ID

    Returns:
        按时间顺序排列的事件列表；轨迹文件不存在时返回空列表
    """
    path = TRACE_DIR / f"{thread_id}.jsonl"
    if not path.exists():
        return []
    events = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def count_events(events: list[dict], event: str) -> int:
    """统计指定类型事件的数量（评测断言的基础工具）"""
    return sum(1 for e in events if e.get("event") == event)


def filter_events(events: list[dict], event: str) -> list[dict]:
    """筛选出指定类型的全部事件（评测断言的基础工具）"""
    return [e for e in events if e.get("event") == event]
