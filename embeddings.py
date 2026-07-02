"""
A minimal, reliable wrapper around FastEmbed's TextEmbedding model.

We use this instead of langchain_community's FastEmbedEmbeddings wrapper
because that wrapper has a known bug under certain LangChain/Pydantic
version combinations where it fails to store its internal model instance
(resulting in "'NoneType' object has no attribute 'embed'"). Talking to
the fastembed library directly avoids that bug entirely.
"""
from typing import List
from fastembed import TextEmbedding
from langchain_core.embeddings import Embeddings


class LocalFastEmbedEmbeddings(Embeddings):
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self._model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [vec.tolist() for vec in self._model.embed(texts)]

    def embed_query(self, text: str) -> List[float]:
        return list(self._model.embed([text]))[0].tolist()
