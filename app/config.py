import os

from dotenv import load_dotenv

# 加载项目根目录的 .env 文件，把其中的变量读入环境
load_dotenv()

# LLM 配置：DeepSeek 使用 OpenAI 兼容接口，只需替换 base_url
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
