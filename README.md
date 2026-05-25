# UK Immigration Rules RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about UK immigration rules using official GOV.UK data. The project scrapes all 105 sections of the UK Immigration Rules via the GOV.UK Content API, chunks and embeds them, and provides a conversational Q&A interface powered by a local LLM through Ollama. It demonstrates a practical approach to building a domain-specific legal Q&A system that runs entirely on a laptop.

## Features

- Scrapes all 105 sections of the UK Immigration Rules from the GOV.UK Content API
- Cleans raw HTML into plain text using BeautifulSoup
- Chunks documents with configurable word-level splitting and overlap
- Two RAG implementations:
  - **Notebook pipeline** (LangChain): FAISS vector store, HuggingFace embeddings, `ConversationalRetrievalChain` with conversation memory
  - **Standalone script** (Haystack): `InMemoryDocumentStore`, dual retrieval (BM25 + sentence-transformer embedding), custom `OllamaGenerator` component
- Sentence embeddings via `all-MiniLM-L6-v2` (lightweight, CPU-friendly)
- Local LLM inference via Ollama (default model: `mistral:instruct`)
- Source attribution with links back to official GOV.UK pages
- Interactive CLI with debug mode to inspect retrieved documents

## Project Structure

```
UK_visa_RAG/
├── notebooks/
│   ├── Data_extraction.ipynb       # Scrapes GOV.UK Content API, builds corpus, embeds with FAISS
│   └── UK_immigration_chatbot.py   # Haystack-based RAG pipeline (standalone script)
├── data/
│   ├── immigration_rules.csv       # Extracted rules (105 sections, ~2.8 MB)
│   └── immigration_rules.json      # Same data in JSON format (~3.0 MB)
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Tech Stack

| Component | Library |
|---|---|
| Data extraction | requests, BeautifulSoup, pandas |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector store (notebook) | FAISS via LangChain |
| Vector store (script) | Haystack `InMemoryDocumentStore` |
| RAG orchestration | LangChain (notebook), Haystack (script) |
| LLM inference | Ollama (local) |
| Notebook | Jupyter |

## Getting Started

### Prerequisites

1. **Python 3.10+**

2. **Ollama** -- install and start the server:
   ```bash
   # Install Ollama (see https://ollama.com for platform-specific instructions)
   curl -fsSL https://ollama.com/install.sh | sh

   # Pull a model
   ollama pull mistral:instruct

   # Start the server (if not already running)
   ollama serve
   ```

### Installation

```bash
git clone https://github.com/ansingh16/UK_visa_RAG.git
cd UK_visa_RAG
pip install -r requirements.txt
```

### Usage

**Option 1: Notebook pipeline (recommended for exploration)**

1. Open and run `notebooks/Data_extraction.ipynb` to scrape GOV.UK, build the corpus, create FAISS embeddings, and query the chatbot interactively.

**Option 2: Standalone Haystack script**

1. First ensure data has been extracted (run the notebook above, or verify `data/immigration_rules.csv` exists).
2. Run the chatbot:
   ```bash
   python notebooks/UK_immigration_chatbot.py
   ```
3. Type questions at the prompt. Special commands:
   - `update` -- refresh the knowledge base from GOV.UK
   - `debug <question>` -- show retrieved documents before answering
   - `quit` -- exit

## Example Interaction

**User:** "Can I apply for a skilled worker visa without sponsorship?"

**Assistant:**

> According to the Immigration Rules, applicants for a Skilled Worker visa must usually have a certificate of sponsorship from a UK employer. The points-based system requires a job offer from an approved sponsor (20 points), a job at the appropriate skill level (20 points), English language proficiency (10 points), and meeting the salary threshold (20+ points).
>
> Source: Immigration Rules Appendix Skilled Worker

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

Immigration Rules content is sourced from the GOV.UK Content API and is Crown copyright.
