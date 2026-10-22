from openai import OpenAI
from pydantic import BaseModel, ValidationError

from app.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL


def get_llm_client() -> OpenAI:
    """创建 LLM 客户端（DeepSeek 的 OpenAI 兼容接口）

    换模型供应商时只需改 .env 里的 LLM_BASE_URL 和 LLM_MODEL，
    业务代码不用动——这就是兼容接口的好处。
    """
    if not LLM_API_KEY:
        raise RuntimeError(
            "未配置 LLM_API_KEY：请在项目根目录创建 .env 文件并填入 API Key"
        )
    return OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)


def chat(messages: list[dict], temperature: float = 0.7) -> str:
    """调用大模型进行一轮对话，返回回复文本

    messages 标准 OpenAI 格式：[{"role": "system"/"user"/"assistant", "content": "..."}]
    temperature 控制随机性：0 最稳定（适合评估打分），1 更发散（适合出题）。
    """
    client = get_llm_client()
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=temperature,
    )
    # 取第一个候选的回复内容
    return response.choices[0].message.content


def chat_with_tools(messages: list, tools: list, temperature: float = 0.7):
    """带工具声明的对话调用，返回完整的 assistant 消息对象

    与 chat() 的区别：模型可能不直接回答，而是返回带 tool_calls 的消息
    （想调用哪个工具、传什么参数）。调用方需要执行工具、把结果以
    role=tool 消息回填进 messages 再次调用——这就是 Agent 循环。
    """
    client = get_llm_client()
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        tools=tools,
        temperature=temperature,
    )
    # 返回完整消息对象：含 content / tool_calls / role 等字段
    return response.choices[0].message


def _clean_json(text: str) -> str:
    """清洗 LLM 输出：去掉 markdown 代码围栏，得到纯 JSON 文本

    模型常无视指令输出 ```json ... ``` 围栏，这里做兜底清洗。
    （原先在 service.py，M9 起上移到 LLM 层供 chat_structured 使用）
    """
    text = text.strip()
    if text.startswith("```"):
        # 去掉首行（```json）与结尾的 ``` 围栏
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    return text.strip()


def chat_structured(
    messages: list[dict],
    model_cls: type[BaseModel],
    temperature: float = 0.3,
    max_retries: int = 2,
):
    """调用 LLM 并解析成 Pydantic 模型；校验失败把错误回填重试（M9 场景①：自我纠错）

    核心思想：错误不是终点而是反馈信号——模型输出了坏 JSON 时，
    把坏输出原文和失败原因喂回去，让它"看着自己的错误"修正，
    而不是让图节点直接崩溃。重试耗尽仍失败则抛出最后一次异常
    （熔断后的降级交给上层，M11 系统化）。

    Args:
        messages: 标准 OpenAI 格式消息
        model_cls: 目标 Pydantic 模型类（如 PlannerDirective / EvaluationResult）
        temperature: 采样温度
        max_retries: 校验失败后的最大重试次数（总调用 = 1 + max_retries）

    Returns:
        model_cls 的校验通过实例
    """
    convo = list(messages)
    last_error: ValidationError | None = None
    for _ in range(max_retries + 1):
        raw = chat(messages=convo, temperature=temperature)
        try:
            return model_cls.model_validate_json(_clean_json(raw))
        except ValidationError as exc:
            last_error = exc
            first = exc.errors()[0]
            # 回填坏输出 + 纠错指令：不悄悄丢弃失败尝试，让模型看到自己错在哪
            convo.append({"role": "assistant", "content": raw})
            convo.append({
                "role": "user",
                "content": (
                    f"你上次的输出无法解析为目标 JSON（校验错误：{first.get('msg')}）。"
                    "请重新输出：严格只输出符合格式要求的 JSON，"
                    "不要任何解释、前后缀或代码围栏。"
                ),
            })
    raise last_error  # type: ignore[misc]
