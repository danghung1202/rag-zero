
from __future__ import annotations
from abc import ABC, abstractmethod

from langchain_openai.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings

class LLMRag:

    def __init__(
        self, 
        loader,
        splitter, 
        model, 
        vector_db
    ) -> None:
        self.loader = loader
        self.splitter = splitter
        self.model = model
        self.vector_db = vector_db



class LLMRagBuilder:
    def __init__(self) -> None:
        self.loader = None
        self.splitter = None
        self.ai_model = None
        self.vector_db = None

    def set_loader(self, loader):
        self.loader = loader
        return self

    def set_splitter(self, splitter):
        self.splitter = splitter
        return self

    def set_model(self, model):
        self.model = model
        return self

    def set_vector_db(self, vector_db):
        self.vector_db = vector_db
        return self

    def build(self):
        return LLMRag(self.loader, self.splitter, self.model, self.vector_db)


if __name__ == '__main__':
    builder = LLMRagBuilder()
    openai_rag = builder.set_model('ai model + embedding')\
                        .set_loader('loader: pdf loader, csv, html, json etc')\
                        .set_vector_db('vector db: load, create, update: (chroma, pipecone...)')\
                        .set_splitter('how splitter ')\
                        .build()
    
    model = openai_rag.model
    