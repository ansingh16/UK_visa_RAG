"""Text chunking utilities for immigration rules documents."""

import re
from typing import List, Dict

import pandas as pd


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(
    text: str,
    max_tokens: int = 480,
    overlap: int = 60,
) -> List[str]:
    """Word-level chunking with overlap.

    Splits *text* into chunks of approximately *max_tokens* words,
    stepping forward by ``max_tokens - overlap`` words each iteration.
    """
    words = normalize_whitespace(text).split()
    chunks: List[str] = []
    i = 0
    step = max(1, max_tokens - overlap)
    while i < len(words):
        chunk = words[i : i + max_tokens]
        if not chunk:
            break
        chunks.append(" ".join(chunk))
        i += step
    return chunks


def build_corpus(
    df: pd.DataFrame,
    max_tokens: int = 480,
    overlap: int = 60,
    text_col: str = "text",
    title_col: str = "title",
    url_col: str = "web_url",
) -> pd.DataFrame:
    """Chunk every row of *df* and return a flat corpus DataFrame.

    Returns columns: doc_title, web_url, chunk_id, chunk_text
    """
    records: List[Dict] = []
    for _, row in df.iterrows():
        chunks = chunk_text(row[text_col], max_tokens=max_tokens, overlap=overlap)
        for j, ch in enumerate(chunks):
            records.append({
                "doc_title": row[title_col],
                "web_url": row[url_col],
                "chunk_id": j,
                "chunk_text": ch,
            })
    return pd.DataFrame(records)
