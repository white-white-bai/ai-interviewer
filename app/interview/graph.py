"""M2：用 LangGraph 重写面试流程（与 M1 手写状态机 app/interview/state_machine.py 并存对比）

M1 的痛点 → M2 的解法：
- 状态存在内存字典里，重启即丢 → checkpointer 每执行一步自动把完整状态写入 SQLite
- 追问/下一题/结束的分支写在 if/else 里 → add_conditional_edges 声明式注册
- 没有"停在这里等外部输入"的机制 → interrupt() 官方人工介入点

图的拓扑（一次完整面试的执行轨迹）：
START → plan_topic（M5：规划官决定下一个考察方向）
      → generate_question（M5：面试官按规划指令定向出题）
      → ask_candidate（interrupt 暂停，等候选人提交答案）
      → run_code（M6：沙箱真实执行回答中的代码块）
      → evaluate（M5：评估官独立人设打分；M7：回答先过安全清洗）
      → manage_memory（M4：滑窗外的早期问答滚动压缩进摘要）
      → 条件边：followup/next → 回到 plan_topic（循环）
               finish        → approval（M8：结束面试属不可逆动作，先过人工审批）
                               批准 → summarize → END
                               驳回 → 回 plan_topic 继续面试

三个 Agent（规划官/面试官/评估官）不互相喊话，
通过共享 state 的 directive 字段传递结构化消息。
"""

import operator
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Annotated, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.runnables import RunnableConfig  # 节点第二参数必须标这个类型，LangGraph 才会注入 config
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from app import llm, sandbox
from app.interview import agents, fallbacks, tracing
from app.interview.memory import format_memory, manage_memory
from app.interview.prompts import (
    EVALUATE_PROMPT,
    QUESTION_WITH_TOOLS_PROMPT,
    SUMMARY_PROMPT,
)
from app.interview.schemas import EvaluationResult, NextAction, QAItem
from app.interview.tools import TOOLS, execute_tool, extract_text_tool_calls
from app.security import sanitize_answer

# 检查点数据库：LangGraph 每个步骤都会把状态快照写进这个文件
# 注意：这是 LangGraph 自己的"存档库"，与 M3 要做的业务数据库是两回事
_CHECKPOINT_DB = Path(__file__).resolve().parent.parent.parent / "checkpoints.sqlite"

_conn = sqlite3.connect(_CHECKPOINT_DB, check_same_thread=False)
checkpointer = SqliteSaver(_conn)


def _tid(config: RunnableConfig | None) -> str:
    """从 LangGraph 运行配置里取 thread_id（M10 轨迹记录用）

    LangGraph 节点函数的第二参数标注为 RunnableConfig 时自动注入；
    thread_id 就是我们 make_config 时传的"会话主键"。
    """
    return ((config or {}).get("configurable") or {}).get("thread_id", "")


class InterviewGraphState(TypedDict):
    """图的共享状态：所有节点读写同一份 state，LangGraph 负责合并与存档

    Annotated[list[dict], operator.add] 是"reducer"声明：
    节点返回 {"history": [新一条]} 时，LangGraph 用列表相加合并而不是覆盖——
    这就是历史问答自动累加的秘密（M1 里要手动 append）。
    """

    candidate_name: str
    position: str
    current_question: str | None
    last_answer: str | None
    last_score: int | None
    last_comment: str | None
    action: str | None
    history: Annotated[list[dict], operator.add]
    followup_count: int
    round_count: int
    summary: str | None
    # M4 记忆字段：摘要 + 进度水位线（前 memory_upto 轮已并入摘要）
    memory_summary: str | None
    memory_upto: int
    # M5 多 Agent 字段：规划官写给面试官的定向出题指令（PlannerDirective 字典）
    directive: dict | None
    # M6 沙箱字段：候选人代码块的真实运行结果（本轮没有代码块则为 None）
    code_result: str | None
    # M8 人工审批字段：评估官的 finish 决策是否被人工批准
    approved: bool | None


def _format_history(history: list[dict]) -> str:
    """全量拼接历史问答（M4 起仅供 summarize 终评使用——总结需要完整信息）

    出题提示词已改用 memory.format_memory（摘要 + 滑窗），
    这里保留是因为最终评语必须覆盖每一轮，属于一次性开销。
    """
    if not history:
        return "（暂无，这是第一个问题）"
    lines = []
    for i, qa in enumerate(history, 1):
        lines.append(f"{i}. 问：{qa['question']}\n   答：{qa['answer']}（{qa['score']} 分）")
    return "\n".join(lines)


# Agent 循环最大工具轮数：防止模型无限调工具烧钱（成本熔断雏形，M11 系统化）
MAX_TOOL_ROUNDS = 3


def plan_topic(state: InterviewGraphState, config: RunnableConfig) -> dict:
    """考察规划节点（M5 新增）：多 Agent 协作的第一环

    规划官读记忆文本（摘要+滑窗），决定下一个考察知识点，
    结果经 Pydantic 校验后写入 state["directive"]——
    Agent 之间不直接对话，图的共享状态就是协作介质。

    M10：规划结果同步落轨迹（评测脚本据此断言"每一题都经规划"）。
    M11：LLM 重试耗尽仍失败时降级到兜底考点池——面试不中断，
    降级事实落轨迹（plan_fallback 事件）供审计。
    """
    thread_id = _tid(config)
    try:
        directive = agents.plan_next_topic(state, thread_id=thread_id)
    except Exception as exc:  # 降级要兜一切意外（网络/解析/未知），不能让图崩溃
        print(f"[规划官·降级] LLM 不可用（{type(exc).__name__}），使用兜底考点池")
        directive = fallbacks.fallback_directive(state.get("round_count", 0))
        tracing.log_event(thread_id, "plan_fallback", reason=str(exc)[:120],
                          topic=directive.topic)
    print(f"[规划官] 下一考点：{directive.topic}（{directive.guidance}）")
    tracing.log_event(thread_id, "plan", topic=directive.topic, guidance=directive.guidance)
    return {"directive": directive.model_dump()}


def generate_question(state: InterviewGraphState, config: RunnableConfig) -> dict:
    """出题节点（M5 升级为面试官 Agent）：按规划官指令定向出题

    Agent 循环（function calling 标准套路，M3-b）：
    1. 工具声明随消息发给模型
    2. 模型要么直接给出问题文本（循环结束），要么返回 tool_calls
    3. 执行工具，结果以 role=tool 消息回填，回到第 1 步

    M5 变化：messages[0] 换成面试官角色的 system prompt，
    user 消息里注入规划官的 directive——出题从"自由发挥"变"按计划执行"。

    M10：工具调用/伪装接管/兜底全程落轨迹（评测脚本据此断言
    工具调用与纠错行为是否符合预期）。
    M11：LLM 重试耗尽仍失败（或工具层意外）时降级到兜底题池——
    面试不中断，降级事实落轨迹（generate_fallback 事件）供审计。

    出题用高温度（0.8），鼓励发散覆盖更多知识点。
    """
    thread_id = _tid(config)
    directive = state.get("directive") or {}
    directive_text = (
        f"知识点：{directive.get('topic', '自由选择')}；"
        f"指令：{directive.get('guidance', '无')}"
    )
    user_content = QUESTION_WITH_TOOLS_PROMPT.format(
        position=state["position"],
        candidate_name=state["candidate_name"],
        directive=directive_text,
        memory_text=format_memory(state),
    )
    messages: list = [
        {"role": "system", "content": agents.INTERVIEWER_SYSTEM.format(position=state["position"])},
        {"role": "user", "content": user_content},
    ]
    question = ""
    try:
        for _ in range(MAX_TOOL_ROUNDS):
            message = llm.chat_with_tools(messages, tools=TOOLS, temperature=0.8)
            if message.tool_calls:
                # 标准路径：模型返回了结构化 tool_calls，按协议回填
                # 协议要求：先把 assistant 的 tool_calls 消息入列，再回填对应结果
                messages.append(message)
                for call in message.tool_calls:
                    # M10 起双轨：print 给人看（终端观察），log_event 给程序看（评测断言）
                    print(f"[面试官·工具调用] {call.function.name}({call.function.arguments})")
                    result = execute_tool(call.function.name, call.function.arguments)
                    print(f"[面试官·工具结果] {result[:200]}")
                    tracing.log_event(thread_id, "tool_call", name=call.function.name,
                                      arguments=call.function.arguments, result=result[:200])
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call.id,  # 结果必须与请求一一对应
                        "content": result,
                    })
                continue
            content = (message.content or "").strip()
            fake_calls = extract_text_tool_calls(content)
            if fake_calls:
                # M9 场景③：DeepSeek 偶发把 <tool_call> 按训练模板格式写进正文，
                # 兼容层解析不出 tool_calls（M3-b 伏笔兑现）——解析后代为执行，
                # 结果以 user 消息回填（不伪造 role=tool 消息，避免协议校验失败）
                messages.append(message)
                observed = []
                for call in fake_calls:
                    print(f"[面试官·文本伪装工具调用→已接管] {call['name']}({call['arguments']})")
                    result = execute_tool(call["name"], call["arguments"])
                    print(f"[面试官·工具结果] {result[:200]}")
                    observed.append(f"{call['name']}({call['arguments']}) → {result}")
                tracing.log_event(thread_id, "fake_tool_call",
                                  calls=[c["name"] for c in fake_calls], results=observed)
                messages.append({
                    "role": "user",
                    "content": (
                        "系统提示：检测到你把工具调用写在正文 <tool_call> 标签里，"
                        "本接口只认标准 tool_calls 字段。已代为执行，结果如下：\n"
                        + "\n".join(observed)
                        + "\n请基于以上结果继续；若已能出题，只输出问题本身。"
                    ),
                })
                continue
            # 既无 tool_calls 也无伪装调用：message.content 就是最终问题
            question = content
            break
        if not question:
            # 兜底：达到工具轮数上限仍没拿到最终问题，撤掉工具声明直接要答案
            # 注意 M5 后 messages[0] 是 system 人设，必须保留 [:2] 再撤工具
            question = llm.chat(messages=messages[:2], temperature=0.8).strip()
            tracing.log_event(thread_id, "fallback_no_tools", question=question[:120])
    except Exception as exc:  # 降级要兜一切意外，不能让图崩溃
        print(f"[面试官·降级] LLM/工具不可用（{type(exc).__name__}），使用兜底题池")
        question = fallbacks.fallback_question(state.get("round_count", 0))
        tracing.log_event(thread_id, "generate_fallback", reason=str(exc)[:120])
    return {
        "current_question": question,
        "round_count": state.get("round_count", 0) + 1,
    }


def ask_candidate(state: InterviewGraphState) -> dict:
    """人工介入节点：interrupt() 把图"暂停"在这里（M2 核心机制）

    - 首次执行到 interrupt()：图立刻停下，当前状态存入检查点，
      payload（当前题目）可通过 get_state 读出，进程甚至可以直接退出
    - 外部调用 invoke(Command(resume=答案)) 后：图从这个断点"原地复活"，
      interrupt() 的返回值就是候选人提交的答案
    """
    answer = interrupt({"question": state["current_question"]})
    return {"last_answer": answer}


# 代码块提取：匹配 ```python ... ``` 或无语言标注的 ``` ... ```（DOTALL 跨行）
_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def run_code(state: InterviewGraphState, config: RunnableConfig) -> dict:
    """代码沙箱节点（M6 新增）：候选人回答里的代码块真正跑一遍

    Agent 决策依据的升级：评估官不再猜"这段代码能不能跑"，
    而是拿到真实运行结果（输出/报错/超时）再打分——
    LLM 心算运行结果经常出错，沙箱给的是事实。

    M10：沙箱执行结果落轨迹（评测脚本据此断言"有代码就有实测"）。

    本轮回答没有代码块时显式把 code_result 置回 None：
    该字段留在 state 里会跨轮残留，必须每轮清理。
    """
    thread_id = _tid(config)
    answer = state.get("last_answer") or ""
    matches = list(_CODE_BLOCK_RE.finditer(answer))
    if not matches:
        tracing.log_event(thread_id, "sandbox", status="no_code", blocks=0)
        return {"code_result": None}  # 无代码块：清理旧值并空转通过
    # 取最后一个代码块：候选人常先用小片段讲解，最终答案代码通常放在最后
    code = matches[-1].group(1)
    print(f"[沙箱] 检测到 {len(matches)} 个代码块，执行最后一个（超时 {sandbox.DEFAULT_TIMEOUT} 秒）...")
    result = sandbox.run_python(code)
    label = "成功" if result["ok"] else ("超时强杀" if result["timed_out"] else "报错")
    preview = (result["stdout"] or result["stderr"]).strip()[:120]
    print(f"[沙箱] 执行{label}：{preview}")
    tracing.log_event(thread_id, "sandbox", status=label, blocks=len(matches), preview=preview)
    return {"code_result": sandbox.format_result(result)}


def evaluate(state: InterviewGraphState, config: RunnableConfig) -> dict:
    """评估节点（M5 升级为评估官 Agent）：低温度（0.2）打分 + Pydantic 结构化校验 + 记录本轮问答

    评估官有独立的 system 人设（agents.EVALUATOR_SYSTEM），
    与出题的面试官分离——避免"出题者自己评估自己"的角色混淆。
    M6：若沙箱跑出了真实运行结果，作为证据追加进评估输入——
    不改共享的 EVALUATE_PROMPT 模板（M1 的 service 也在用它）。
    M7：候选人回答是不可信的外部数据，注入提示词前先过 sanitize_answer
    （截断 + 数据/指令分离标记）；history 里存的仍是原始回答——
    延续 M4 的"无损存储、有损注入"原则。
    M9：解析交给 chat_structured——评估官偶发输出坏 JSON 时自动回填
    错误重试，不再让 evaluate 节点直接崩溃。
    M10：评分结果落轨迹（评测脚本据此断言分数/动作分布与重试预算）。
    M11：LLM 重试耗尽仍失败时降级为中性默认分（5 分 + 降级标记），
    不惩罚候选人、不卡死流程，降级事实落轨迹（evaluate_fallback）。
    """
    thread_id = _tid(config)
    # M7：清洗不可信回答，评估输入用的是带数据标记的版本
    safe_answer = sanitize_answer(state["last_answer"])
    if state.get("code_result"):
        safe_answer = (
            f"{safe_answer}\n\n"
            f"【沙箱实测】候选人代码块的真实运行结果（以此为准，不要凭记忆猜测代码输出）：\n"
            f"{state['code_result']}"
        )
    try:
        result = llm.chat_structured(
            messages=[
                {"role": "system", "content": agents.EVALUATOR_SYSTEM},
                {
                    "role": "user",
                    "content": EVALUATE_PROMPT.format(
                        position=state["position"],
                        question=state["current_question"],
                        answer=safe_answer,  # 用清洗后的版本（此前误传原始回答，沙箱证据没真正进提示词）
                    ),
                },
            ],
            model_cls=EvaluationResult,
            temperature=0.2,
            trace_thread_id=thread_id,  # M10：坏 JSON 重试同步落轨迹
        )
    except Exception as exc:  # 降级要兜一切意外，不能让图崩溃
        print(f"[评估官·降级] LLM 不可用（{type(exc).__name__}），按中性分处理")
        result = fallbacks.fallback_evaluation()
        tracing.log_event(thread_id, "evaluate_fallback", reason=str(exc)[:120])
    qa = QAItem(
        question=state["current_question"],
        answer=state["last_answer"],
        score=result.score,
        comment=result.comment,
        created_at=datetime.now(),
    )
    tracing.log_event(thread_id, "evaluate", score=result.score,
                      action=result.action.value, comment=result.comment)
    updates: dict = {
        "last_score": result.score,
        "last_comment": result.comment,
        "action": result.action.value,
        "history": [qa.model_dump()],  # history 声明了 operator.add，这里只给增量
    }
    if result.action is NextAction.FOLLOWUP:
        updates["followup_count"] = state.get("followup_count", 0) + 1
    return updates


def manage_memory_node(state: InterviewGraphState, config: RunnableConfig) -> dict:
    """记忆管理节点（M12 复盘补齐）：M4 压缩 + M10 轨迹 + M11 降级三合一

    M12 复盘时发现的两个缺口：
    1. M4 压缩事件从未落轨迹（机制覆盖断言缺一环）
    2. M11 降级漏了本节点——压缩 LLM 彻底失败会让图崩溃

    降级语义：跳过本轮压缩（摘要暂不更新、滑窗照常扩大）——
    记忆层降级 = 延迟压缩，只影响 token 效率，不影响正确性。
    """
    try:
        updates = manage_memory(state)
    except Exception as exc:  # 降级要兜一切意外，不能让图崩溃
        print(f"[记忆·降级] 压缩失败（{type(exc).__name__}），本轮跳过（滑窗照常）")
        tracing.log_event(_tid(config), "memory_fallback", reason=str(exc)[:120])
        updates = {}
    if updates:
        tracing.log_event(_tid(config), "memory",
                          upto=updates["memory_upto"], summary_chars=len(updates["memory_summary"]))
    return updates


def route_after_evaluate(state: InterviewGraphState) -> str:
    """条件边路由函数：返回值决定走哪条边

    返回值必须与 add_conditional_edges 注册的映射键一致
    （"followup" / "next" / "finish"），相当于 M1 转移表的声明式版本。
    """
    return state["action"]


def approval(state: InterviewGraphState, config: RunnableConfig) -> dict:
    """人工审批节点（M8 新增）：评估官的 finish 决策执行前必须过人

    结束面试并生成终评是不可逆动作——终评一出，面试即成定论。
    interrupt() 在这里第二次暂停图，把决策权交给外部人类：
    - 外部 Command(resume=True)  → 批准，条件边走向 summarize 正常收尾
    - 外部 Command(resume=False) → 驳回，条件边回到 plan_topic 继续面试
      （M2 的检查点保证驳回后现场不丢，图原地"倒车回主路"）

    M10：审批请求与人工决定都落轨迹（评测脚本据此断言
    "finish 从未被绕过审批直接执行"——不可逆动作的审计证据）。

    教学点：interrupt 不只能"等输入"，还能做"高风险决策的人工把关"——
    Agent 可以自主建议，但不可逆动作的执行权在人类手里。
    """
    tracing.log_event(_tid(config), "approval_requested",
                      round_count=state.get("round_count", 0),
                      last_score=state.get("last_score"))
    approved = bool(interrupt({
        "event": "finish_approval",
        "message": "评估官建议结束面试并生成终评报告（不可逆动作），请审批",
        "round_count": state.get("round_count", 0),
        "last_score": state.get("last_score"),
    }))
    tracing.log_event(_tid(config), "approval", approved=approved)
    return {"approved": approved}


def route_after_approval(state: InterviewGraphState) -> str:
    """审批结果路由：批准去生成终评，驳回回规划官继续面试"""
    return "summarize" if state["approved"] else "plan_topic"


def summarize(state: InterviewGraphState, config: RunnableConfig) -> dict:
    """总结节点（M8 起审批通过后才执行）：生成整场面试的评语，随后图走到 END

    M11：LLM 不可用时降级为统计式报告（基于 history 拼接，不调 LLM），
    终评"有交代"，降级事实落轨迹（summary_fallback）。
    """
    try:
        report = llm.chat(
            messages=[{
                "role": "user",
                "content": SUMMARY_PROMPT.format(
                    position=state["position"],
                    candidate_name=state["candidate_name"],
                    history_text=_format_history(state.get("history", [])),
                ),
            }],
            temperature=0.4,
        )
    except Exception as exc:  # 降级要兜一切意外，不能让图崩溃
        print(f"[终评·降级] LLM 不可用（{type(exc).__name__}），输出统计式报告")
        report = fallbacks.fallback_summary(state)
        tracing.log_event(_tid(config), "summary_fallback", reason=str(exc)[:120])
    tracing.log_event(_tid(config), "summary", round_count=state.get("round_count", 0))
    return {"summary": report.strip(), "current_question": None}


def build_interview_graph():
    """组装并编译面试图：节点 + 边 + 条件边 + 检查点持久化

    add_edge 是固定流转（等价于 M1 转移表里的一行），
    add_conditional_edges 是"看状态选路"（等价于 M1 里的 if/else 分支）。
    """
    builder = StateGraph(InterviewGraphState)
    builder.add_node("plan_topic", plan_topic)  # M5：规划官 Agent
    builder.add_node("generate_question", generate_question)
    builder.add_node("ask_candidate", ask_candidate)
    builder.add_node("run_code", run_code)  # M6：代码沙箱节点
    builder.add_node("evaluate", evaluate)
    builder.add_node("manage_memory", manage_memory_node)  # M4：记忆管理节点（M12 补轨迹+降级）
    builder.add_node("approval", approval)  # M8：人工审批节点
    builder.add_node("summarize", summarize)

    builder.add_edge(START, "plan_topic")
    builder.add_edge("plan_topic", "generate_question")
    builder.add_edge("generate_question", "ask_candidate")
    builder.add_edge("ask_candidate", "run_code")
    builder.add_edge("run_code", "evaluate")
    builder.add_edge("evaluate", "manage_memory")
    # 条件边从 manage_memory 挂出：history 已更新完毕，摘要也已滚动，
    # followup/next 都回到规划官——每一道新题都先经规划再出题；
    # finish 不再直达总结，而是先去审批节点（M8：不可逆动作过人）
    builder.add_conditional_edges(
        "manage_memory",
        route_after_evaluate,
        {"followup": "plan_topic", "next": "plan_topic", "finish": "approval"},
    )
    # M8：审批结果分岔——批准收尾，驳回回到规划官继续面试
    builder.add_conditional_edges(
        "approval",
        route_after_approval,
        {"summarize": "summarize", "plan_topic": "plan_topic"},
    )
    builder.add_edge("summarize", END)
    # compile 时挂上 checkpointer：之后每一步自动写档，支持暂停/恢复/读档
    return builder.compile(checkpointer=checkpointer)


# 模块级单例：应用生命周期内复用同一个图和同一个检查点连接
interview_graph = build_interview_graph()


def make_config(thread_id: str) -> dict:
    """构造 LangGraph 运行配置

    thread_id 是检查点里的"会话主键"（对应 M1 的 session_id）：
    同一 thread_id 的所有调用共享并续写同一份存档。
    """
    return {"configurable": {"thread_id": thread_id}}
