from abc import ABC, abstractmethod

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel


class LLMPort(ABC, BaseModel):
    @abstractmethod
    def get_llm_client(self) -> BaseChatModel:
        pass
