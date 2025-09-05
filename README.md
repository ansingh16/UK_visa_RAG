
# UK Immigration Rules Assistant

This project demonstrates how to build a **Retrieval-Augmented Generation (RAG) chatbot** using the **UK Immigration Rules** as a knowledge base. The assistant can answer user queries by retrieving the most relevant sections of the official GOV.UK guidance and generating natural language responses.

The goal is to showcase **practical AI/NLP skills** for real-world applications, such as **customer support, legal tech, or policy Q\&A systems**, all deployable on a laptop.

---

## 🚀 Features

* 📂 **Data ingestion**: Scrapes or retrieves Immigration Rules content via the **GOV.UK Content API**.
* 🔍 **Semantic search**: Embeds rules text with `sentence-transformers` and stores in a vector database (FAISS/Chroma).
* 💬 **RAG pipeline**: Retrieves top-k relevant chunks and feeds them to an LLM for context-aware answers.
* 🌐 **Interactive UI**: A simple **Streamlit app** provides a chat interface.
* 📑 **Source attribution**: Responses include links to the original GOV.UK rules.
* 💻 **Laptop-friendly**: Runs with lightweight models (`all-MiniLM-L6-v2` for embeddings, OpenAI API or local LLM).

---

## 📂 Project Structure

```
ai-immigration-assistant/
│── data/                     # Scraped or API-fetched immigration rules
│── embeddings/               # Saved vector database
│── app.py                    # Streamlit chatbot app
│── ingest.py                 # Script to fetch & embed rules
│── rag_pipeline.py            # Core RAG logic
│── requirements.txt          # Dependencies
│── README.md                 # Documentation
```

---


## 📊 Example Interaction

**User:** "Can I apply for a skilled worker visa without sponsorship?"
**Assistant:**

> According to the Immigration Rules, applicants for a Skilled Worker visa must usually have a certificate of sponsorship from a UK employer.
>
> 🔗 Source: [Immigration Rules Part 5](https://www.gov.uk/guidance/immigration-rules/part-5)

---

## 🛠️ Future Extensions

* 🔁 Expand beyond Immigration Rules (e.g., tax, finance, healthcare regulations).
* 🌍 Multilingual support for broader accessibility.
* 🤖 Integrate with WhatsApp/Slack for real-world deployment.
* 📊 Analytics dashboard: track most common queries.

---

## 📜 License

This project is for educational purposes. Content is sourced via the GOV.UK Content API and is © Crown copyright.

---

