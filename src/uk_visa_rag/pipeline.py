"""End-to-end RAG pipeline for UK immigration questions."""

import logging
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

from uk_visa_rag.chunker import build_corpus
from uk_visa_rag.embeddings import (
    build_faiss_index,
    get_embedding_model,
    load_faiss_index,
    save_faiss_index,
)
from uk_visa_rag.retriever import FAISSRetriever
from uk_visa_rag.scraper import ImmigrationRulesScraper

logger = logging.getLogger(__name__)

QA_TEMPLATE = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are an assistant helping with UK Immigration Rules.
Use the provided context ONLY. Ignore clauses marked "DELETED".
Give clear, concise answers in plain English.
If rules are complex, summarize them in bullet points.
At the end, cite the relevant section titles.

Context:
{context}

Question: {question}

Answer:""",
)


class RAGPipeline:
    """Orchestrates scraping, chunking, embedding, retrieval, and generation.

    Usage::

        pipeline = RAGPipeline.from_data_dir("data")
        answer, sources = pipeline.query("What visa do I need to work in the UK?")
    """

    def __init__(
        self,
        retriever: FAISSRetriever,
        llm_model: str = "mistral:instruct",
        ollama_url: str = "http://localhost:11434",
        memory_window: int = 10,
    ):
        self.retriever = retriever
        self.llm = OllamaLLM(model=llm_model, base_url=ollama_url)
        self.memory = ConversationBufferWindowMemory(k=memory_window)
        self._chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever.as_langchain_retriever(),
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": QA_TEMPLATE},
            return_source_documents=True,
        )

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_data_dir(
        cls,
        data_dir: str | Path = "data",
        index_dir: str | Path | None = None,
        llm_model: str = "mistral:instruct",
        chunk_size: int = 480,
        chunk_overlap: int = 60,
        **kwargs,
    ) -> "RAGPipeline":
        """Build a pipeline from scraped data on disk.

        If *index_dir* exists, load the FAISS index from there;
        otherwise build a fresh index from the CSV and (optionally) save it.
        """
        data_dir = Path(data_dir)
        embedding_model = get_embedding_model()

        if index_dir and Path(index_dir).exists():
            logger.info("Loading existing FAISS index from %s", index_dir)
            index = load_faiss_index(index_dir, embedding_model)
        else:
            csv_path = data_dir / "immigration_rules.csv"
            if not csv_path.exists():
                raise FileNotFoundError(
                    f"{csv_path} not found. Run the scraper first: "
                    "ImmigrationRulesScraper().fetch_and_save()"
                )
            df = pd.read_csv(csv_path)
            corpus = build_corpus(df, max_tokens=chunk_size, overlap=chunk_overlap)
            logger.info("Building FAISS index from %d chunks", len(corpus))
            index = build_faiss_index(corpus, embedding_model)

            if index_dir:
                save_faiss_index(index, index_dir)
                logger.info("Saved FAISS index to %s", index_dir)

        retriever = FAISSRetriever(index)
        return cls(retriever=retriever, llm_model=llm_model, **kwargs)

    @classmethod
    def from_scratch(
        cls,
        data_dir: str | Path = "data",
        llm_model: str = "mistral:instruct",
        chunk_size: int = 480,
        chunk_overlap: int = 60,
        **kwargs,
    ) -> "RAGPipeline":
        """Scrape GOV.UK, build the corpus and index, then return a pipeline."""
        scraper = ImmigrationRulesScraper()
        df = scraper.fetch_and_save(data_dir)
        corpus = build_corpus(df, max_tokens=chunk_size, overlap=chunk_overlap)
        embedding_model = get_embedding_model()
        index = build_faiss_index(corpus, embedding_model)
        retriever = FAISSRetriever(index)
        return cls(retriever=retriever, llm_model=llm_model, **kwargs)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, question: str) -> tuple[str, List[Dict[str, Any]]]:
        """Ask a question and return (answer, sources).

        Each source dict has keys: title, url, content (excerpt).
        """
        result = self._chain.invoke({"question": question})
        answer = result.get("answer", "")
        sources = []
        for doc in result.get("source_documents", []):
            sources.append({
                "title": doc.metadata.get("title", ""),
                "url": doc.metadata.get("url", ""),
                "content": doc.page_content[:300],
            })
        return answer, sources

    def search(self, question: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks without generating an answer."""
        return self.retriever.retrieve(question, top_k=top_k)

    def clear_memory(self) -> None:
        """Reset conversation history."""
        self.memory.clear()
