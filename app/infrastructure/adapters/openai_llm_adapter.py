from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from app.config.settings import Settings
from app.domain.ports.llm_port import LLMPort


class OpenAiLlmAdapter(LLMPort):
    def get_llm_client(self) -> BaseChatModel:
        settings = Settings()
        return ChatOpenAI(
            model=settings.OPENAI_MODEL,
            api_key=settings.OPENAI_API_KEY,
            temperature=0,
        )
