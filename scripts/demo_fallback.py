"""M11 演示：降级兜底——LLM 挂了，系统"优雅变差"而不是崩溃

三幕：
幕一 网络层：桩注入让 API 调用先超时 2 次再成功 → 观察指数退避重试
     自愈；再让它永远超时 → 观察重试耗尽后异常上抛（不吞异常，
     降级责任交给节点层）。全程零真实 LLM 调用（纯桩）。
幕二 节点层：真实图 + 桩让 LLM 永远失败 → 规划/出题/评估/终评依次
     降级（兜底考点池/兜底题池/中性 5 分/统计式报告），面试照常走完
     ——不崩溃、不卡死、不惩罚候选人。
幕三 轨迹审计：读回幕二的轨迹，断言"降级都留痕 + 面试正常收尾"
     ——优雅变差必须是可审计的变差。

全程约 1 分钟（几乎不调真实 LLM，只有沙箱子进程和 sleep）。
"""

import sys
import types
import uuid
from pathlib import Path

# 把项目根目录加入模块搜索路径（Windows 直接运行脚本时 sys.path 不含根目录）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
from langgraph.types import Command
from openai import APITimeoutError

from app import llm
from app.interview import tracing
from app.interview.graph import interview_graph, make_config

_TIMEOUT = APITimeoutError(request=httpx.Request("POST", "https://api.test/v1/chat"))


def _fake_client(create_fn) -> types.SimpleNamespace:
    """构造一个假的 OpenAI 客户端（只需要 chat.completions.create 一条路径）

    Args:
        create_fn: 替身函数，签名同 client.chat.completions.create(**kwargs)
    """
    return types.SimpleNamespace(
        chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(create=create_fn)
        )
    )


def _fake_response(content: str) -> types.SimpleNamespace:
    """构造一个假的 API 响应（只需要 choices[0].message.content 一条路径）"""
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(
            message=types.SimpleNamespace(content=content, tool_calls=None),
        )]
    )


def act1_network_retry() -> None:
    """幕一：网络抖动退避重试自愈 + 重试耗尽上抛（纯桩，零真实调用）"""
    real = llm.get_llm_client

    # A. 前两次超时、第三次成功 → 退避重试自愈
    counter = {"n": 0}

    def flaky_create(**kwargs):
        """替身：模拟不稳定网络——前 2 次抛超时，之后恢复正常"""
        counter["n"] += 1
        if counter["n"] <= 2:
            raise _TIMEOUT
        return _fake_response("重试后成功")

    llm.get_llm_client = lambda: _fake_client(flaky_create)
    out = llm.chat(messages=[{"role": "user", "content": "hi"}])
    print(f"A. 调用 3 次才成功（2 次超时被退避重试自愈）：{out}")

    # B. 永远超时 → 重试耗尽，异常原样上抛（网络层不吞异常）
    def dead_create(**kwargs):
        """替身：模拟服务彻底不可用——永远抛超时"""
        raise _TIMEOUT

    llm.get_llm_client = lambda: _fake_client(dead_create)
    try:
        llm.chat(messages=[{"role": "user", "content": "hi"}])
    except APITimeoutError:
        print("B. 重试 3 次全部失败，异常上抛——降级责任交给节点层")
    finally:
        llm.get_llm_client = real  # 恢复真实客户端（幕二自己会再打桩）


def act2_graph_degrade(thread_id: str) -> None:
    """幕二：LLM 永远失败，真实图全链路降级走完面试（真实 LLM 零调用）"""
    def dead_create(**kwargs):
        """替身：模拟 DeepSeek 服务彻底宕机——任何调用都抛超时"""
        raise _TIMEOUT

    llm.get_llm_client = lambda: _fake_client(dead_create)
    try:
        # 启动面试：规划降级 → 出题降级 → 停在 ask_candidate 等回答
        interview_graph.invoke(
            {"candidate_name": "小测", "position": "Python 后端"},
            make_config(thread_id),
        )
        q1 = interview_graph.get_state(make_config(thread_id)).values["current_question"]
        print(f"\n第 1 题（兜底题池）：{q1[:60]}…")
        print("[候选人] 回答一个示例答案（正常提交，沙箱照常执行）")
        interview_graph.invoke(Command(resume="print('示例回答')"), make_config(thread_id))
        values = interview_graph.get_state(make_config(thread_id)).values or {}
        print(f"[第 1 轮评估（降级）] {values.get('last_score')} 分 | {values.get('last_comment')[:40]}…")
        # 评估降级固定 NEXT → 回到规划官（继续降级）→ 出题降级 → 停在第 2 题
        q2 = interview_graph.get_state(make_config(thread_id)).values["current_question"]
        print(f"第 2 题（兜底题池轮换）：{q2[:60]}…")
        # 注入 finish 收尾（M8 手法）：批准 → 终评降级 → END
        interview_graph.update_state(
            make_config(thread_id), {"action": "finish"}, as_node="manage_memory",
        )
        interview_graph.invoke(None, make_config(thread_id))
        interview_graph.invoke(Command(resume=True), make_config(thread_id))
        state = interview_graph.get_state(make_config(thread_id))
        print(f"\n[终评（降级·统计式）]\n{(state.values or {}).get('summary')}")
        print(f"\n[收尾] next={state.next}（空元组 = 已到 END，全程没有崩溃）")
    finally:
        # 恢复真实客户端：演示脚本不打持久补丁
        import importlib
        importlib.reload(llm)


def act3_audit(thread_id: str) -> int:
    """幕三：轨迹审计——降级都留痕 + 面试正常收尾（返回失败断言数）"""
    events = tracing.load_trace(thread_id)
    kinds = [e["event"] for e in events]
    checks = [
        ("规划降级有留痕", "plan_fallback" in kinds),
        ("出题降级有留痕", "generate_fallback" in kinds),
        ("评估降级有留痕", "evaluate_fallback" in kinds),
        ("终评降级有留痕", "summary_fallback" in kinds),
        ("面试正常收尾", kinds[-1] == "summary" and kinds.count("summary") == 1),
        ("降级轮也有完整评估记录", kinds.count("evaluate") == 1),
    ]
    print(f"\n===== 幕三：轨迹审计（共 {len(events)} 条事件）=====")
    failures = 0
    for name, ok in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        failures += 0 if ok else 1
    print("\n降级事件明细：")
    for e in events:
        if e["event"].endswith("_fallback"):
            print(f"  {e['ts']} {e['event']} reason={e.get('reason', '')[:60]}")
    return failures


def main() -> None:
    """三幕连演：网络重试 → 全链路降级 → 轨迹审计"""
    print("===== 幕一：网络层退避重试（纯桩）=====")
    act1_network_retry()
    thread_id = uuid.uuid4().hex[:8]
    print(f"\n===== 幕二：LLM 彻底宕机，图全链路降级（thread_id={thread_id}）=====")
    print(f"轨迹文件：traces/{thread_id}.jsonl")
    act2_graph_degrade(thread_id)
    failures = act3_audit(thread_id)
    if failures == 0:
        print("\n===== M11 演示通过：降级可审计，系统优雅变差而非崩溃 =====")
    else:
        print(f"\n===== M11 演示失败：{failures} 条断言未过 =====")
        sys.exit(1)


if __name__ == "__main__":
    main()
