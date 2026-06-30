"""Vector database wrapper around ChromaDB (local, persistent, no server).

Stores each comment as: document text + embedding + metadata
(video_id, author, likes, sentiment, entities, topic). Metadata enables the
"metadata retrieval" strategy required by the rubric (e.g. filter to positive
comments about a given entity before semantic ranking).
"""
from __future__ import annotations

from typing import Dict, List, Optional

from config import EMBEDDING_MODEL, VECTOR_DB_DIR

COLLECTION = "youtube_comments"


class VectorStore:
    def __init__(self, persist_dir: str = VECTOR_DB_DIR):
        import chromadb
        from chromadb.utils import embedding_functions

        self._client = chromadb.PersistentClient(path=persist_dir)
        self._ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        self.collection = self._client.get_or_create_collection(
            name=COLLECTION, embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, ids: List[str], docs: List[str], metadatas: List[Dict]):
        # Chroma rejects None / list metadata values -> coerce to primitives.
        clean = [{k: _scalar(v) for k, v in m.items()} for m in metadatas]
        self.collection.add(ids=ids, documents=docs, metadatas=clean)

    def query(self, text: str, k: int = 5,
              where: Optional[Dict] = None) -> List[Dict]:
        res = self.collection.query(query_texts=[text], n_results=k, where=where)
        out = []
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            out.append({"text": doc, "metadata": meta, "distance": dist})
        return out

    def all_documents(self) -> List[str]:
        return self.collection.get()["documents"]

    def count(self) -> int:
        return self.collection.count()


def _scalar(v):
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v if v is not None else ""
    if isinstance(v, (list, tuple)):
        return ", ".join(map(str, v))
    return str(v)
