from openai import OpenAI

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
