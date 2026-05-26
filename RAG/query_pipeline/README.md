# Query Pipeline — Restaurant Grand Chennai

Companion to the ingestion pipeline. Takes customer queries, retrieves
relevant chunks from Qdrant, and generates responses via **Cerebras** (Llama-3.3-70B).

---

## Architecture

```
Customer query
     │
     ▼
┌──────────────────┐
│  query_session   │  Persistent state: allergens, veg prefs, history
└──────────────────┘
     │
     ▼
┌──────────────────┐
│  query_intent    │  Intent classifier — routing + filter signals
│                  │  type: menu | booking | info | policy | general
│                  │  veg_only, exclude_allergens, max_price, spice_pref
└──────────────────┘
     │
     ▼
┌──────────────────────────────────────────┐
│  query_retriever  (Qdrant Hybrid Search) │
│                                          │
│  Dense prefetch  → BAAI/bge-base (768d) │ top-20 semantic candidates
│  Sparse prefetch → Qdrant/BM25          │ top-20 keyword candidates
│  RRF fusion      → merged top-5         │ best of both
│  Metadata filter → pre-filter by intent │ BEFORE vector search
└──────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────┐
│  query_llm  (Cerebras Llama-3.3-70B)    │
│                                          │
│  ~1,800 tok/s — first token in ~100ms   │
│  Streaming SSE support                  │
│  Allergen-safe system prompt            │
└──────────────────────────────────────────┘
     │
     ▼
   Response
```

---

## Setup

### 1. Install dependencies
```bash
pip install fastembed qdrant-client requests python-dotenv fastapi uvicorn
```

### 2. Add Cerebras API key to `.env`
```
CEREBRAS_API_KEY=your_key_here
```
Get a free key at: https://cloud.cerebras.ai

### 3. Ensure Qdrant storage exists
The ingestion pipeline must have been run first:
```bash
python main.py --skip-refinement
```
This creates `./qdrant_storage` with the `restaurant_grand_chennai` collection.

---

## Usage

### Interactive CLI
```bash
cd query_pipeline
python query_pipeline.py
```

### Single query
```bash
python query_pipeline.py --query "What vegetarian starters do you have?"
```

### Run test suite
```bash
python query_pipeline.py --test
```

### With debug output
```bash
python query_pipeline.py --verbose
```

### Start the REST + WebSocket API
```bash
python query_api.py
# API docs: http://localhost:8000/docs
```

### API example
```bash
# Blocking
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"message": "What veg dishes do you have?", "session_id": null}'

# Streaming (SSE)
curl -N -X POST http://localhost:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "I am allergic to gluten. What can I eat?"}'
```

---

## File Structure

```
query_pipeline/
├── query_config.py     # Cerebras + Qdrant config
├── query_intent.py     # Intent classifier (regex-first)
├── query_retriever.py  # Qdrant hybrid search + filter builder
├── query_llm.py        # Cerebras LLM call (streaming)
├── query_session.py    # Multi-turn session manager
├── query_pipeline.py   # Master orchestrator + CLI
└── query_api.py        # FastAPI REST + WebSocket server
```

---

## Cerebras vs. Other LLMs

| Provider            | Model             | Speed (tok/s) | First token |
|---------------------|-------------------|---------------|-------------|
| **Cerebras**        | Llama-3.3-70B     | ~1,800        | ~100ms      |
| Groq                | Llama-3.3-70B     | ~280          | ~200ms      |
| OpenAI              | GPT-4o            | ~80           | ~500ms      |
| Anthropic           | Claude Sonnet     | ~100          | ~400ms      |

For a restaurant chatbot where customers expect instant answers, Cerebras
is the fastest option available at this quality level.

---

## Safety Features

1. **Allergen hard-filter**: Allergen exclusions are applied as Qdrant
   `must_not` filters — dishes are excluded BEFORE vector search, not after.
   Semantic similarity cannot override an allergen exclusion.

2. **Session persistence**: Declared allergens persist across all turns in
   a session. A customer who says "I'm allergic to dairy" in turn 1 will
   never see dairy dishes in turns 2, 3, 4...

3. **No hallucination guardrail**: System prompt instructs the LLM to only
   state prices and allergens from retrieved context. If context is empty,
   the LLM is instructed to say "I don't have that information."

4. **Fallback retrieval**: If a filtered search returns 0 results, the
   pipeline retries with allergen-only filters (dropping other filters)
   to ensure safety filters are never dropped while preference filters may
   be relaxed.
