# AI 面试官：M1-M12 架构复盘

> 按 12 步学习路线边做边学构建的 AI 面试官系统。本文档是全程复盘：
> 每一层解决什么问题、关键文件在哪、踩过哪些反直觉的坑、离生产还差什么。

## 一、系统全景

```
                     ┌─────────────────────────────────────────┐
  HTTP 层 (M1)       │  FastAPI：/interview/* /questions 等     │
                     │  Pydantic 请求校验（M7 入口层在此）        │
                     └───────────────┬─────────────────────────┘
                                     │
                     ┌───────────────▼─────────────────────────┐
  图编排层 (M2)      │  LangGraph StateGraph + interrupt        │
                     │  SqliteSaver 检查点（跨进程可恢复）        │
                     │  条件边：followup/next/finish + 审批路由   │
                     └───────────────┬─────────────────────────┘
                                     │
        ┌──────────────┬─────────────┼──────────────┬─────────────┐
        ▼              ▼             ▼              ▼             ▼
  规划官(M5)      面试官(M3/M5)   沙箱(M6)      评估官(M5/M9)  记忆(M4)
  plan_topic     generate_      run_code      evaluate      manage_memory
  决定考察方向    question       真实执行       打分+动作     滑窗+滚动摘要
  (Pydantic)     Agent工具循环   代码块         (低温度)      (无损存储
   directive →      ↑工具层(M3-b)+伪装接管(M9)                  有损注入)
   共享 state ──────┘
        │
        ▼
  支撑层：LLM 封装(M1/M9/M11) │ 安全(M7) │ 轨迹(M10) │ 兜底题池(M11)
```

**协作模型**：多 Agent 之间不互相喊话，共享 state 就是协作介质
（规划官写 `directive`，面试官读）；不可逆动作（finish）必须过
`interrupt` 人工审批（M8）。

## 二、里程碑地图

| 里程碑 | 解决的问题 | 关键文件 | 演示脚本 | commit |
|---|---|---|---|---|
| M1 手写状态机 | 最小可用：LLM 封装 + 状态转移表 + CRUD | app/llm.py, app/interview/{state_machine,service}.py | demo_interview.py | a0a76b2 |
| M2 LangGraph | 状态持久化（重启不丢）、声明式分支、interrupt 人工介入 | app/interview/graph.py | demo_graph.py | d87e4f7 |
| M3 SQLite+工具 | 业务数据落库；Agent 通过 function calling 查题库；MCP 跨进程暴露工具 | app/{db,storage}.py, app/interview/tools.py, mcp_server.py | demo_mcp_client.py | c4677cc, a0f5604 |
| M4 记忆压缩 | 长面试 token 爆炸：滑窗 + 滚动摘要（无损存储、有损注入） | app/interview/memory.py | demo_memory.py | 09362de |
| M5 多 Agent | 出题要发散、评估要严格、规划要稳定——三个人设三个温度 | app/interview/agents.py | demo_multi_agent.py | 43b0557 |
| M6 代码沙箱 | 评估依据从"猜代码输出"升级为"真实运行结果" | app/sandbox.py | demo_sandbox.py | b0bf5ec |
| M7 安全 | 提示词注入、越权工具、超长输入：数据与指令分离 + 最小权限 | app/security.py | demo_security.py | 7d6c107 |
| M8 人工审批 | 不可逆动作（生成终评）前的人类把关 | graph.approval | demo_approval.py | 41464f3, d49f887 |
| M9 自我纠错 | 坏 JSON 回填重试；`<tool_call>` 文本伪装接管；工具报错自愈 | app/llm.py chat_structured, tools.extract_text_tool_calls | demo_self_correction.py | b5c15f8 |
| M10 轨迹评测 | print 人肉观察 → JSONL 轨迹 + 程序化断言（可接 CI） | app/interview/tracing.py | demo_tracing.py | f3c43b2 |
| M11 降级兜底 | 网络退避重试 + 四节点降级路径：优雅变差而非崩溃 | app/llm.py, app/interview/fallbacks.py | demo_fallback.py | d0cf4ab |
| M12 复盘 | 全景演示（一条轨迹串起全部机制）+ 本文档 | scripts/demo_full_interview.py, docs/REVIEW.md | demo_full_interview.py | — |

## 三、韧性三层（M9/M11 分工）

| 层 | 触发条件 | 手段 | 失败后 |
|---|---|---|---|
| 网络层 (M11) | 超时/连接错误/限流/5xx | 指数退避重试 1s→2s ×3 | 原样上抛，不吞异常 |
| 内容层 (M9) | 结构化输出坏 JSON | 坏输出+错误原因回填重试 ×2 | 抛最后异常，交节点层 |
| 节点层 (M11) | LLM 彻底不可用 | 兜底考点池/题池/中性 5 分/统计式报告 | 系统继续跑，`*_fallback` 轨迹留痕 |

降级设计原则：产物结构化类型不变（下游无感知）；不惩罚候选人
（评估降级给 5 分不是 0 分）；动作固定 NEXT（FOLLOWUP 打转、
FINISH 未评先终）；一切降级可审计。

## 四、反直觉坑清单（最容易再踩的）

1. **LangGraph 节点第二参数必须标 `RunnableConfig`**——标 `dict`
   它就不注入 config，按单参数节点调用报 missing argument（M10 踩过）。
2. **interrupt 有重放语义**——resume 后节点函数整个重新执行，
   interrupt() 之前的副作用会重复（M10 轨迹里 approval_requested 记两次）。
   轨迹=原始事实不去重，评测端理解语义。
3. **M5 之后 messages[0] 是 system 人设**——工具轮数兜底撤工具时
   必须 `messages[:2]` 而不是 `messages[0]`，否则把 user 需求也撤了。
4. **history 用 `Annotated[list, operator.add]`**——节点返回增量
   `[qa.model_dump()]`，返回全量会翻倍。
5. **无代码块时显式置 `code_result=None`**——state 字段跨轮残留，
   不清理会拿上一轮的沙箱结果评这一轮。
6. **DeepSeek 会把 `<tool_call>` 当正文输出**——`tool_calls` 检查
   防不住文本伪装，需要正则提取接管（M3-b 伏笔，M9 兑现）；
   代执行结果用 user 消息回填，不伪造 `role=tool`（协议校验会炸）。
7. **`json.loads` 直接解析模型参数会崩图**——M7 权限层顺手兜了
   畸形 JSON。
8. **LLM 决策时机不可控**——评估官何时 finish 有强随机性，演示
   脚本不要赌模型行为，用 `update_state(as_node=...)` 注入决策保证
   确定性（M8 实测两段 6 轮全 followup）。
9. **依赖必须锁版本**——`pip install mcp` 装到 2.x API 破坏性变更。
10. **Windows cmd 管道/重定向中文乱码**——先 `set PYTHONUTF8=1`
    再跑；PowerShell 的 Tee-Object 反而会按 GBK 解码产生乱码。
11. **沙箱实测前 M6 有个真 bug**——拼好的沙箱证据变量没传进
    `format()`（传的还是原始回答），证据根本没进提示词；M7 重写
    时才暴露修复。
12. **降级覆盖要全图清点**——M11 收尾时规划/出题/评估/终评四节点
    都接了，但记忆压缩节点漏了，M12 复盘才补上。复盘的价值实证。

## 五、遗留清单（生产化方向）

- **沙箱**：当前是本机子进程（超时强杀+临时目录）；生产应换
  Docker 容器/独立低权限账号/网络隔离/内存 CPU 限额（Windows 无 rlimits）。
- **检查点**：SQLite 单文件 → Redis/Postgres 后端，支持多实例部署。
- **观测**：自建 JSONL → LangSmith/Langfuse 托管（可视化、检索、
  在线评测），注意轨迹出域的数据合规。
- **评测**：demo_tracing 的断言接 CI；补 LLM-as-judge 的回答质量
  评测集（轨迹评测验"行为"，质量评测验"效果"）。
- **并发**：多候选人并发压测（threading.Lock 的轨迹写、SQLite
  锁竞争、限流）。
- **安全**：sanitize 是白名单思路的雏形，生产补：输出侧审查
  （评估意见也进终评，同样不可信）、工具调用频次熔断、审计日志留存策略。
- **成本**：出题/评估/压缩的 token 计量与预算熔断（MAX_TOOL_ROUNDS
  是雏形，可系统化为 per-thread 预算）。

## 六、运行指南

```cmd
:: 起服务（/docs 交互验证）
uvicorn app.main:app --reload

:: 单机制演示（按里程碑顺序）
python scripts\demo_interview.py      :: M1 手写状态机
python scripts\demo_graph.py          :: M2 LangGraph
python scripts\demo_mcp_client.py     :: M3 MCP
python scripts\demo_memory.py         :: M4 记忆
python scripts\demo_multi_agent.py    :: M5 多 Agent
python scripts\demo_sandbox.py        :: M6 沙箱
python scripts\demo_security.py       :: M7 安全
python scripts\demo_approval.py       :: M8 审批
python scripts\demo_self_correction.py:: M9 纠错
python scripts\demo_tracing.py        :: M10 轨迹评测（--eval-only <id> 复核）
python scripts\demo_fallback.py       :: M11 降级兜底

:: M12 全景：一条真实面试轨迹串起全部机制 + 机制覆盖断言
python scripts\demo_full_interview.py
```
