"""Retrieval logic wrapping the FAISS vector store."""

from typing import List, Dict, Any

from langchain_community.vectorstores import FAISS


class FAISSRetriever:
    """Thin wrapper around a FAISS index for document retrieval."""

    def __init__(self, index: FAISS, top_k: int = 5):
        self.index = index
        self.top_k = top_k

    def retrieve(self, query: str, top_k: int | None = None) -> List[Dict[str, Any]]:
        """Return the top-k most relevant chunks for *query*.

        Each result dict contains 'content', 'title', 'url', and 'score'.
        """
        k = top_k or self.top_k
        docs_with_scores = self.index.similarity_search_with_score(query, k=k)
        results = []
        for doc, score in docs_with_scores:
            results.append({
                "content": doc.page_content,
                "title": doc.metadata.get("title", ""),
                "url": doc.metadata.get("url", ""),
                "score": float(score),
            })
        return results

    def as_langchain_retriever(self, **kwargs):
        """Return a LangChain-compatible retriever."""
        return self.index.as_retriever(search_kwargs={"k": self.top_k, **kwargs})
