"""Embedding and vector store utilities."""

from pathlib import Path
from typing import List, Optional

import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embedding_model(model_name: str = DEFAULT_MODEL) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=model_name)


def build_faiss_index(
    corpus: pd.DataFrame,
    embedding_model: Optional[HuggingFaceEmbeddings] = None,
    text_col: str = "chunk_text",
    title_col: str = "doc_title",
    url_col: str = "web_url",
) -> FAISS:
    """Build a FAISS vector store from a corpus DataFrame."""
    if embedding_model is None:
        embedding_model = get_embedding_model()

    texts = corpus[text_col].tolist()
    metadatas = [
        {"title": row[title_col], "url": row[url_col]}
        for _, row in corpus.iterrows()
    ]
    return FAISS.from_texts(texts=texts, embedding=embedding_model, metadatas=metadatas)


def save_faiss_index(index: FAISS, path: str | Path) -> None:
    index.save_local(str(path))


def load_faiss_index(
    path: str | Path,
    embedding_model: Optional[HuggingFaceEmbeddings] = None,
) -> FAISS:
    if embedding_model is None:
        embedding_model = get_embedding_model()
    return FAISS.load_local(
        str(path), embedding_model, allow_dangerous_deserialization=True
    )
