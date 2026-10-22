"""面试模块的数据模型（与题库 app/models.py 分离，避免职责混淆）"""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

from app.interview.state_machine import InterviewState
from app.security import MAX_ANSWER_LENGTH


class NextAction(str, Enum):
    """评估后 AI 决定的下一步动作"""

    FOLLOWUP = "followup"  # 围绕当前题目追问
    NEXT = "next"          # 换下一题
    FINISH = "finish"      # 结束面试


class QAItem(BaseModel):
    """一轮完整的问答记录"""

    question: str
    answer: str
    score: int = Field(ge=0, le=10)
    comment: str = Field(description="AI 评估意见")
    created_at: datetime


class EvaluationResult(BaseModel):
    """LLM 评估输出的结构化结果（Pydantic 校验不通过 = 模型输出不合规）"""

    score: int = Field(ge=0, le=10, description="回答得分 0-10")
    comment: str = Field(description="评估意见，一两句话")
    action: NextAction = Field(description="下一步动作")


class PlannerDirective(BaseModel):
    """考察规划官的结构化输出（M5 多 Agent 协作的"消息协议"）

    规划官把决策写进这个结构 → 图存入 state["directive"] → 出题 Agent 读取：
    多 Agent 之间不直接对话，通过共享状态的字段传递消息。
    """

    topic: str = Field(description="下一个考察的知识点")
    guidance: str = Field(description="给面试官的一句话出题指令，如：围绕生成器在异常时的行为深入追问")


class InterviewSession(BaseModel):
    """一场面试会话的完整状态（M1 内存存储，M3 持久化到 SQLite）"""

    id: str
    candidate_name: str
    position: str
    state: InterviewState = InterviewState.CREATED
    current_question: str | None = None
    history: list[QAItem] = []
    followup_count: int = 0


class InterviewStartRequest(BaseModel):
    """开始面试的请求体"""

    candidate_name: str = Field(min_length=1, description="候选人姓名")
    position: str = Field(min_length=1, description="面试岗位方向，如 Python 后端")


class InterviewAnswerRequest(BaseModel):
    """提交回答的请求体"""

    # M7 入口层防御：超长报文在 HTTP 边界就被 422 拒绝，不给恶意大报文
    # 进入业务逻辑的机会（绕过 HTTP 的内部调用由 sanitize_answer 清洗层兜底）
    answer: str = Field(
        min_length=1,
        max_length=MAX_ANSWER_LENGTH,
        description="候选人的回答内容",
    )


class GraphInterviewSnapshot(BaseModel):
    """LangGraph 版面试会话快照（数据来自检查点 get_state，M2 新增）"""

    thread_id: str = Field(description="LangGraph 线程 ID，等同于 M1 的 session_id")
    status: str = Field(description="waiting_answer=等待回答 / finished=已结束")
    next_node: list[str] = Field(description="图接下来要执行的节点；空列表=已到 END")
    candidate_name: str | None = None
    position: str | None = None
    current_question: str | None = None
    last_score: int | None = None
    last_comment: str | None = None
    action: str | None = None
    summary: str | None = None
    followup_count: int = 0
    round_count: int = 0
    history: list[QAItem] = []
    # M4 记忆字段：早期问答滚动摘要（history 仍为全量原文，二者并存）
    memory_summary: str | None = None
