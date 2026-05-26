"""
query_pipeline/query_config.py
───────────────────────────────
Central config for the query pipeline.
Mirrors ingestion config but adds Cerebras settings.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Cerebras (fast LLM for response generation) ────────────────
# WHY Cerebras: ~2,000 tokens/sec on Llama 3.1-70B, far faster
# than OpenAI/Anthropic for real-time restaurant chat queries.
# Free tier available at cloud.cerebras.ai → API key → .env
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "")
CEREBRAS_API_URL = "https://api.cerebras.ai/v1/chat/completions"

# Model options (from Cerebras docs):
#   "llama-3.3-70b"   → best quality, ~1,800 tok/s
#   "llama3.1-8b"     → faster, good for simple Q&A
CEREBRAS_MODEL   = "llama3.1-8b"

# ── Qdrant ─────────────────────────────────────────────────────
COLLECTION_NAME = "restaurant_grand_chennai"
QDRANT_PATH     = r"E:\ministosAI_RAG\qdrant_storage"   # same path as ingestion

# ── Embedding models (MUST match ingestion pipeline exactly) ───
DENSE_MODEL  = "BAAI/bge-base-en-v1.5"
SPARSE_MODEL = "Qdrant/bm25"

# ── Retrieval settings ──────────────────────────────────────────
TOP_K_PREFETCH = 20   # candidates per vector type before RRF fusion
TOP_K_FINAL    = 5    # final results returned to LLM after fusion

# ── Guardrail settings ──────────────────────────────────────────
MAX_HISTORY_TURNS   = 6    # keep last N turns in context window
MAX_CONTEXT_CHARS   = 4000 # max chars of retrieved text sent to LLM
ALLERGEN_SAFE_MODE  = True # hard-filter allergens even if user un-sets
