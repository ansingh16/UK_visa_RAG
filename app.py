"""Gradio chat UI for the UK immigration rules RAG chatbot.

Thin front end over ``uk_visa_rag.RAGPipeline``: it takes a question, runs the
retrieval-augmented chain against a local Ollama model, and shows the answer with
links back to the GOV.UK sections it drew on.

Run locally (needs Ollama running and the scraped corpus in ``data/``)::

    pip install -e ".[app]"
    python app.py

Configuration is read from the environment so the same file can run locally or on
a host with a different model/endpoint:

    UK_VISA_DATA_DIR    corpus directory (default: data)
    UK_VISA_INDEX_DIR   FAISS index directory; built from the corpus if absent
                        (default: faiss_index)
    UK_VISA_LLM_MODEL   Ollama model name (default: mistral:instruct)
    OLLAMA_URL          Ollama server URL (default: http://localhost:11434)
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

import gradio as gr

# Allow running straight from a checkout without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from uk_visa_rag import RAGPipeline  # noqa: E402

DATA_DIR = os.environ.get("UK_VISA_DATA_DIR", "data")
INDEX_DIR = os.environ.get("UK_VISA_INDEX_DIR", "faiss_index")
LLM_MODEL = os.environ.get("UK_VISA_LLM_MODEL", "mistral:instruct")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

EXAMPLE_QUESTIONS = [
    "What visa do I need to work in the UK as a skilled worker?",
    "How long can I stay in the UK on a Standard Visitor visa?",
    "What are the English language requirements for a partner visa?",
    "Can I switch from a Student visa to a Skilled Worker visa from inside the UK?",
    "What is the minimum salary threshold for a Skilled Worker visa?",
]

INTRO = (
    "# UK Immigration Rules Assistant\n"
    "Ask about UK visas and immigration. Answers are grounded in the official "
    "[GOV.UK Immigration Rules](https://www.gov.uk/guidance/immigration-rules) and "
    "each reply links the sections it drew on, so you can check the source.\n\n"
    "*This is an information tool, not legal advice.*"
)


@lru_cache(maxsize=1)
def get_pipeline() -> RAGPipeline:
    """Build the RAG pipeline once and reuse it across requests.

    Cached because loading the embedding model and FAISS index is expensive; the
    first question pays that cost, the rest are fast.
    """
    return RAGPipeline.from_data_dir(
        data_dir=DATA_DIR,
        index_dir=INDEX_DIR,
        llm_model=LLM_MODEL,
        ollama_url=OLLAMA_URL,
    )


def _format_sources(sources: list[dict]) -> str:
    """Render retrieved sources as a deduplicated markdown link list."""
    if not sources:
        return ""
    lines = ["\n\n---\n**Sources**"]
    seen: set[tuple[str, str]] = set()
    for src in sources:
        title = (src.get("title") or "GOV.UK Immigration Rules").strip()
        url = (src.get("url") or "").strip()
        key = (title, url)
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"- [{title}]({url})" if url else f"- {title}")
    return "\n".join(lines)


def _add_user_message(message: str, history: list[dict]) -> tuple[str, list[dict]]:
    """Append the user's turn and clear the input box."""
    if not message.strip():
        return message, history
    return "", history + [{"role": "user", "content": message}]


def _answer(history: list[dict]) -> list[dict]:
    """Run the pipeline on the latest user turn and append the assistant reply."""
    question = history[-1]["content"]
    try:
        answer, sources = get_pipeline().query(question)
        reply = answer + _format_sources(sources)
    except FileNotFoundError:
        reply = (
            "I can't find the immigration-rules corpus. Run the scraper first "
            "(`ImmigrationRulesScraper().fetch_and_save()`) so there's data to search."
        )
    except Exception as exc:  # noqa: BLE001 - surface any backend error in the chat
        reply = (
            f"Something went wrong reaching the model backend ({exc}). "
            "Check that Ollama is running and the configured model is pulled."
        )
    return history + [{"role": "assistant", "content": reply}]


def _reset() -> list[dict]:
    """Clear both the visible chat and the pipeline's conversation memory."""
    try:
        get_pipeline().clear_memory()
    except Exception:  # noqa: BLE001 - nothing to clear if the pipeline never built
        pass
    return []


def build_demo() -> gr.Blocks:
    """Assemble the Gradio interface (kept as a function so tests can import it)."""
    with gr.Blocks(title="UK Immigration Rules Assistant") as demo:
        gr.Markdown(INTRO)
        chatbot = gr.Chatbot(type="messages", height=460, show_copy_button=True)
        question = gr.Textbox(
            label="Your question",
            placeholder="e.g. What visa do I need to work in the UK?",
            autofocus=True,
        )
        with gr.Row():
            ask = gr.Button("Ask", variant="primary")
            clear = gr.Button("Clear conversation")
        gr.Examples(EXAMPLE_QUESTIONS, inputs=question, label="Example questions")

        for trigger in (question.submit, ask.click):
            trigger(
                _add_user_message, [question, chatbot], [question, chatbot], queue=False
            ).then(_answer, chatbot, chatbot)
        clear.click(_reset, None, chatbot, queue=False)

    return demo


if __name__ == "__main__":
    build_demo().queue().launch()
