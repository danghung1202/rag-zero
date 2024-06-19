from __future__ import annotations
from abc import ABC, abstractmethod

from langchain_openai.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings


class AbstractAIRag(ABC):
    def split_to_chunks():
        pass

    def create_prompt(self):
        pass

    @abstractmethod
    def create_model(self):
        pass

    @abstractmethod
    def create_embeddings(self):
        pass

    @abstractmethod    
    def use_vectordb(self):
        pass

    @abstractmethod    
    def __str__(self):
        pass


class OpenAIRag(AbstractAIRag):

    def __init__(self, openai_api_key, temperature) -> None:
        super().__init__()
        self._api_key = openai_api_key
        self._temperature = temperature
        self._model = None
        self._embedding = None

    def create_model(self):
        if not self._model:
            self._model = ChatOpenAI(api_key=self._api_key, temperature=self._temperature)
        return self._model

    def create_embeddings(self, chunk_size=1000):
        if not self._embedding:
            self._embedding = OpenAIEmbeddings(openai_api_key=self._api_key, chunk_size=chunk_size)
        return self._embedding

    def __str__(self):
        return f"{self._api_key}({self._temperature})"


class OpenAIRagCreation:

    def __init__(self):
        self._openAIrag = None

    def __call__(self, openai_api_key, temperature, **_ignored) -> AbstractAIRag:
        if not self._openAIrag:
            self._openAIrag = OpenAIRag(openai_api_key=openai_api_key, temperature=temperature)
        return self._openAIrag


class RagFactory:
    def __init__(self) -> None:
        self._llms = {}

    def register_llm(self, key, llm: AbstractAIRag):
        self._llms[key] = llm

    def use_model(self, key, *args) -> AbstractAIRag:
        llm = self._llms.get(key)
        if not self._llms[key]:
            raise ValueError(key)
        # Invoke to __call__ of LLM creation class
        return llm(*args)


if __name__ == '__main__':
    rag_factory = RagFactory()
    rag_factory.register_llm("OpenAI", OpenAIRagCreation())

    openai_rag = rag_factory.use_model("OpenAI", "open_ai_key", 1000)
    print(openai_rag)
