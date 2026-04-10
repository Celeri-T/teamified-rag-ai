# Teamified RAG AI

A localized Retrieval-Augmented Generation (RAG) system designed to provide context-aware answers. This project implements a full AI pipeline—from PDF ingestion and noise-reduction preprocessing to vector search and LLM generation—all running locally for privacy and offline accessibility.

## 🚀 Features
* **Production-Grade Modular Architecture:** Clean separation of concerns with dedicated modules for document loading, preprocessing, vector storage, and LLM orchestration.
* **Intelligent Preprocessing:** Custom logic to clean the text and filter low-context chunks, ensuring higher retrieval accuracy.
* **Persistent Vector Store:** Utilizes **FAISS** with local disk persistence, allowing for instant startups on subsequent runs by skipping the embedding phase.
* **Optimized Local Inference:** Configured to run **Qwen3-0.6B** locally.
* **Interactive CLI:** A user-friendly command-line interface with cleaned AI responses, free from "model meta-talk."

---

## 🛠️ Setup Instructions

### Environment Preparation
This project requires **Python 3.10+**. It is highly recommended to use a virtual environment.

**1. Clone the repository**
```bash
git clone [https://github.com/Celeri-T/teamified-rag-ai.git](https://github.com/Celeri-T/teamified-rag-ai.git)
cd philippine-history-rag
```

**2. Create and Activate Virtual Environment**

*macOS / Linux*
```bash
python -m venv .venv
source .venv/bin/activate
```

*Windows*
```bash
python -m venv .venv
.venv\Scripts\activate
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```

### Model Preparation
The system uses the **Qwen3-0.6B** model. On the first run, the system will automatically download the model from Hugging Face and cache it locally for offline use.

---

## 🏃 Run Instructions

### Data Ingestion
Ensure your source document (`philippine_history.pdf`) is placed in the local directory.

### Start the Application
Run the main script to initialize the vector database and start the interactive session:
```bash
python run.py
```

### Usage
* Once the **"User Query:"** prompt appears, you can ask questions about Philippine history.

---

## 🧠 Technical Notes: LLM Choice

For this project, **Qwen3-0.6B-Instruct** was chosen as the primary reasoning engine for several strategic reasons:

* **Computational Efficiency:** With only 600M parameters, it serves as an ideal lightweight language model for this assessment.
* **Instruction Adherence:** Qwen3 demonstrates superior ability to follow negative constraints (e.g., "Do not mention the context") compared to larger 1B-3B models, which is crucial for a clean RAG user experience.
* **Local Privacy:** By running the model locally via the `transformers` library, no data is sent to external APIs, ensuring the integrity and privacy of the processed documents.
* **License:** Released under the Apache 2.0 license, allowing for flexible use and modification without the gating requirements often found in larger proprietary models.

---

## 📂 Project Structure

| File | Description |
| :--- | :--- |
| `src/config.py` | Centralized configuration for model paths, chunk sizes, and system constants. |
| `src/document_loader.py` | Handles PDF parsing and initial text chunking. |
| `src/preprocessing.py` | Implements text cleaning and quality-based chunk filtering. |
| `src/vector_store.py` | Manages the FAISS index lifecycle (creation, loading, and saving). |
| `src/llm.py` | Configures the Hugging Face pipeline and `GenerationConfig`. |
| `src/rag_chain.py` | Orchestrates the LangChain Expression Language (LCEL) retrieval chain. |
| `run.py` | The main entry point and interactive controller. |

---

## 📝 Requirements

* `langchain`
* `langchain-huggingface`
* `faiss-cpu` (or `faiss-gpu`)
* `transformers`
* `accelerate`
* `pypdf`
