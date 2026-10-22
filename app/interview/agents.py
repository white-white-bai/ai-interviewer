"""M5：多 Agent 层——把"一个泛化面试官"拆成各司其职的角色 Agent

Agent 的最小构成 = 角色人设（system prompt）+ 任务指令（user prompt）+ 可选工具。

拆分动机：
- 出题要发散（高温）、评估要严格（低温），人格和策略本就不同
- 评估官人设独立，只依据回答内容打分，避免出题者"自己出题自己夸"
- 规划官专职决定"追问还是换题"，出题者只管把题出好（职责单一）

协作方式（LangGraph 语境下的多 Agent）：Agent 之间不互相喊话，
通过图的共享 state 传递结构化消息——规划官写 directive，面试官读。
"""

from app import llm
from app.interview.memory import format_memory
from app.interview.prompts import PLANNER_PROMPT
from app.interview.schemas import PlannerDirective
from app.interview.service import _clean_json

# 三个角色的 system prompt：人设分离是多 Agent 的第一特征
PLANNER_SYSTEM = """你是技术面试的考察规划官。你不出题、不评分，只负责根据面试记忆决定下一个考察知识点，平衡覆盖面与针对性。"""

INTERVIEWER_SYSTEM = """你是「{position}」岗位的资深技术面试官，提问清晰具体、贴合工程实践，一次只问一个问题。"""

EVALUATOR_SYSTEM = """你是严格的技术面试评估官，只依据候选人回答内容本身评分，不受题目难度或提问方式影响。"""


def plan_next_topic(state: dict) -> PlannerDirective:
    """考察规划官 Agent：读记忆文本，决定下一个考察知识点（含追问/换题决策）

    输出经 Pydantic 校验的 PlannerDirective，由图写入 state["directive"]
    传给出题 Agent——这就是多 Agent 之间的"消息协议"。

    Args:
        state: 图的共享状态（至少含 position、history、memory_* 字段）
    """
    raw = llm.chat(
        messages=[
            {"role": "system", "content": PLANNER_SYSTEM},
            {
                "role": "user",
                "content": PLANNER_PROMPT.format(
                    position=state["position"],
                    memory_text=format_memory(state),
                ),
            },
        ],
        temperature=0.3,  # 规划要稳定，不要发散
    )
    return PlannerDirective.model_validate_json(_clean_json(raw))
