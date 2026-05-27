# 🍛 Restaurant Grand Chennai — RAG Chatbot

An AI-powered restaurant assistant built with **LlamaIndex** and **Qdrant**, designed to answer customer questions about menu items, table bookings, allergens, opening hours, and restaurant policies.

The chatbot uses a **Retrieval-Augmented Generation (RAG)** pipeline that ingests restaurant documents (`.docx` files), creates vector embeddings with **HuggingFace BGE**, stores them in a local **Qdrant** vector database, and generates grounded responses using **DeepSeek** as the LLM.

---

## ✨ Features

- **Menu Recommendations** — suggests dishes with prices, veg/non-veg status, and allergen info
- **Table Booking Assistance** — collects booking details and forwards them for confirmation
- **Allergen Safety** — highlights potential allergens from retrieved context
- **Conversational Memory** — remembers context within a chat session
- **Static Routing** — instant responses for greetings, thanks, and goodbyes
- **Hybrid Retrieval** — vector similarity search with similarity cutoff filtering

---

## 🏗️ Architecture

```
docs/ (DOCX files)
  │
  ▼
┌──────────────────────────────────┐
│  Ingestion Pipeline              │
│  parser.py   → StructuralDocxReader (heading-aware DOCX parsing)
│  chunker.py  → SimpleChunker (per-paragraph chunks)
│  metadata.py → Allergen & price extraction
│  indexer.py  → Qdrant vector store indexing
└──────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────┐
│  Qdrant Vector Store             │
│  (local storage: ./qdrant_storage)
└──────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────┐
│  Query Pipeline                  │
│  VectorIndexRetriever (top-k=8)  │
│  SimilarityPostprocessor (≥0.3)  │
│  ResponseSynthesizer (compact)   │
└──────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────┐
│  Chat Engine                     │
│  CondenseQuestionChatEngine      │
│  + ChatMemoryBuffer (4000 tokens)│
└──────────────────────────────────┘
```

---
run   python injection_retrivel.py ---> for injection pipeline
run   python main.py ---> chatbot interface only 

## 📋 Prerequisites

- **Python 3.13+**
- **[uv](https://docs.astral.sh/uv/)** (recommended) or **pip**
- A **DeepSeek API key** ([get one here](https://platform.deepseek.com/))

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd llamaindex_rag
```

### 2. Set Up Environment Variables

Create a `.env` file in the project root:

```env
DEEPSEEK_API_KEY="your-deepseek-api-key"
```

### 3. Install Dependencies

#### Using `uv` (Recommended)

```bash
# Install uv if you don't have it
# See: https://docs.astral.sh/uv/getting-started/installation/

# Create virtual environment and install dependencies
uv sync
```

#### Using `pip`

```bash
python -m venv .venv

# Activate the virtual environment
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# Windows (CMD)
.venv\Scripts\activate.bat
# Linux / macOS
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Ingest Documents (First-Time Setup)

Place your restaurant `.docx` documents in the `docs/` folder. The project includes four documents by default:

| Document | Content |
|---|---|
| `Complete Menu.docx` | Full menu with dishes, prices, and descriptions |
| `Restaurant Grand Chennai.docx` | General restaurant information and facts |
| `TABLE BOOKING SYSTEM.docx` | Table booking policies and procedures |
| `customerexperience.docx` | Customer experience FAQs |

Run the ingestion pipeline to build the vector index:

```bash


This will:
- Parse all `.docx` files with heading-aware structure
- Chunk documents into paragraphs
- Extract metadata (allergens, prices)
- Generate embeddings using `BAAI/bge-base-en-v1.5`
- Store everything in a local Qdrant database at `./qdrant_storage/`

> **Note:** You only need to run ingestion once. The vector store persists in `./qdrant_storage/`.

### 5. Run the Chatbot

```bash
# Using uv
uv run python main.py

# Using pip (with venv activated)
python main.py
```

---

## 💬 Usage

Once the assistant is running, you can interact with it in the terminal:

```
Restaurant assistant ready.

Commands:
  /clear  -> clear conversation memory
  /exit   -> exit assistant

Customer: What vegetarian starters do you have?
