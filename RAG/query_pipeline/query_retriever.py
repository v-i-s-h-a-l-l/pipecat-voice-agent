"""
query_pipeline/query_retriever.py
──────────────────────────────────
WHAT:  Takes a classified intent, builds Qdrant filters,
       embeds the query, runs hybrid search (dense + sparse + RRF),
       and returns ranked context chunks ready for the LLM.

WHY THIS LAYER EXISTS:
  Separation of concerns:
    - query_intent.py  → WHAT does the user want?
    - query_retriever.py → WHERE in Qdrant does the answer live?
    - query_llm.py     → HOW do we phrase the answer?

  The retriever is the bridge between intent and knowledge.

HYBRID SEARCH RECAP:
  1. Dense prefetch  → top-20 by cosine similarity (semantic)
  2. Sparse prefetch → top-20 by BM25 dot product (keyword)
  3. RRF fusion      → merged top-5 by reciprocal rank
  4. Metadata filter → applied BEFORE vector search (pre-filter)

  Pre-filtering is critical: applying filters BEFORE vector search
  reduces the search space so RRF ranks within the valid set only.
  Post-filtering would mean: retrieve irrelevant docs, then discard —
  wasting the entire embedding + retrieval step.

OUTPUT: List of context dicts:
    [
        {
            text:          str,   ← the chunk text to send to LLM
            content_type:  str,
            source_doc:    str,
            item_name:     str | None,
            price:         int | None,
            allergens:     list,
            veg_type:      str | None,
            spice_level:   str | None,
            score:         float,   ← RRF score (for debugging)
        },
        ...
    ]
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 '..', 'pipeline_implementation'))

from query_config import (
    COLLECTION_NAME, QDRANT_PATH, DENSE_MODEL, SPARSE_MODEL,
    TOP_K_PREFETCH, TOP_K_FINAL
)

from fastembed import TextEmbedding, SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter, FieldCondition, MatchValue, Range,
    SparseVector, Prefetch, FusionQuery, Fusion
)


# ── Singleton models (load once, reuse across queries) ─────────
_dense_model  = None
_sparse_model = None
_qdrant_client = None


def get_dense_model():
    global _dense_model
    if _dense_model is None:
        print(f"[Retriever] Loading dense model: {DENSE_MODEL}")
        _dense_model = TextEmbedding(model_name=DENSE_MODEL)
    return _dense_model


def get_sparse_model():
    global _sparse_model
    if _sparse_model is None:
        print(f"[Retriever] Loading sparse model: {SPARSE_MODEL}")
        _sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)
    return _sparse_model


def get_qdrant_client():
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(path=QDRANT_PATH)
        print(f"[Retriever] Connected to Qdrant at {QDRANT_PATH}")
    return _qdrant_client


# ── Query embedder ──────────────────────────────────────────────
def embed_query(query_text: str) -> dict:
    """
    Embeds a customer query into both vector types.
    MUST use identical models as ingestion pipeline.
    """
    dense_model  = get_dense_model()
    sparse_model = get_sparse_model()

    dense_vec    = list(dense_model.embed([query_text]))[0].tolist()
    sparse_result = list(sparse_model.embed([query_text]))[0]
    sparse_idx   = sparse_result.indices.tolist()
    sparse_val   = sparse_result.values.tolist()

    return {
        "dense":          dense_vec,
        "sparse_indices": sparse_idx,
        "sparse_values":  sparse_val,
    }


# ── Filter builder ──────────────────────────────────────────────
def build_filter_from_intent(intent: dict) -> Filter | None:
    """
    Translates a classified intent into a Qdrant Filter.

    SAFETY NOTE on allergens:
    Allergen exclusions are placed in must_not — Qdrant guarantees
    these documents are NEVER returned, regardless of vector score.
    This is a hard filter, not a soft ranking signal.
    This is the correct implementation for food safety.
    """
    must     = []
    must_not = []

    # Veg filter
    if intent.get("veg_only"):
        must.append(FieldCondition(
            key="veg_type",
            match=MatchValue(value="vegetarian")
        ))

    # Price filter
    if intent.get("max_price") is not None:
        must.append(FieldCondition(
            key="price",
            range=Range(lte=intent["max_price"])
        ))

    # Spice filter
    if intent.get("spice_preference"):
        must.append(FieldCondition(
            key="spice_level",
            match=MatchValue(value=intent["spice_preference"])
        ))

    # Document type routing filter
    if intent.get("doc_type_filter"):
        must.append(FieldCondition(
            key="doc_type",
            match=MatchValue(value=intent["doc_type_filter"])
        ))

    # SAFETY: Allergen hard exclusion
    for allergen in intent.get("exclude_allergens", []):
        must_not.append(FieldCondition(
            key="allergens",
            match=MatchValue(value=allergen.lower())
        ))

    if not must and not must_not:
        return None

    return Filter(
        must=must if must else None,
        must_not=must_not if must_not else None,
    )


# ── Hybrid search ───────────────────────────────────────────────
def hybrid_search(query_vectors: dict, filters: Filter | None = None,
                  top_k: int = TOP_K_FINAL) -> list:
    """
    Runs Qdrant hybrid search with Reciprocal Rank Fusion.

    Flow:
      1. Dense prefetch  → semantic candidates (top-20)
      2. Sparse prefetch → keyword candidates (top-20)
      3. RRF fusion      → merged ranking
      4. Return top_k    → final results

    Metadata filter applied at prefetch stage — pre-filtering.
    """
    client = get_qdrant_client()

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            Prefetch(
                query=query_vectors["dense"],
                using="dense",
                limit=TOP_K_PREFETCH,
                filter=filters,
            ),
            Prefetch(
                query=SparseVector(
                    indices=query_vectors["sparse_indices"],
                    values=query_vectors["sparse_values"],
                ),
                using="sparse",
                limit=TOP_K_PREFETCH,
                filter=filters,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k,
        with_payload=True,
    )

    return results.points


# ── Format results for LLM context ─────────────────────────────
def format_context(results: list, max_chars: int = 4000) -> tuple[str, list]:
    """
    Converts Qdrant results into:
      1. A formatted context string for the LLM prompt
      2. A list of source metadata dicts for response attribution

    WHY we use original_text (not refined_text) for the LLM:
    refined_text was enriched for better embedding quality.
    original_text is cleaner and exactly matches what's in the docs.
    The LLM should cite accurate source information.

    WHY max_chars:
    Cerebras Llama-3.3-70B context window is large (128K tokens)
    but sending too much context dilutes the relevant signal.
    4000 chars ≈ ~1000 tokens — focused and effective.
    """
    if not results:
        return "", []

    context_parts = []
    sources       = []
    total_chars   = 0

    for i, point in enumerate(results):
        p    = point.payload
        text = p.get("text", p.get("refined_text", ""))

        # Build a self-contained context block
        block_lines = [f"[Source {i+1}: {p.get('source_doc', 'unknown')}]"]

        if p.get("item_name"):
            block_lines.append(f"Item: {p['item_name']}")
        if p.get("price"):
            block_lines.append(f"Price: ₹{p['price']}")
        if p.get("allergens") is not None:
            allergen_str = ", ".join(p["allergens"]) if p["allergens"] else "None"
            block_lines.append(f"Allergens: {allergen_str}")
        if p.get("veg_type"):
            block_lines.append(f"Type: {p['veg_type']}")
        if p.get("spice_level"):
            block_lines.append(f"Spice: {p['spice_level']}")

        block_lines.append(f"\n{text}")
        block = "\n".join(block_lines)

        if total_chars + len(block) > max_chars:
            break

        context_parts.append(block)
        total_chars += len(block)

        sources.append({
            "source_doc":   p.get("source_doc"),
            "content_type": p.get("content_type"),
            "item_name":    p.get("item_name"),
            "price":        p.get("price"),
            "allergens":    p.get("allergens"),
            "veg_type":     p.get("veg_type"),
            "score":        point.score,
        })

    context_str = "\n\n---\n\n".join(context_parts)
    return context_str, sources


# ── Master retrieve function ────────────────────────────────────
def retrieve(query: str, intent: dict) -> tuple[str, list]:
    """
    Full retrieval pipeline for a single customer query.

    Steps:
      1. Build Qdrant filter from intent
      2. Embed query (dense + sparse)
      3. Hybrid search with filter
      4. Format results as LLM context

    Returns:
      context_str: formatted text to inject into LLM prompt
      sources:     list of source metadata dicts
    """
    # Step 1: Build filter
    qdrant_filter = build_filter_from_intent(intent)

    # Step 2: Embed query
    query_vectors = embed_query(query)

    # Step 3: Hybrid search
    results = hybrid_search(query_vectors, filters=qdrant_filter)

    # Fallback: if filtered search returns nothing, retry without filter
    # (except allergen filters — those MUST stay for safety)
    if not results and qdrant_filter is not None:
        allergen_only_filter = build_filter_from_intent({
            "exclude_allergens": intent.get("exclude_allergens", []),
            "veg_only": False,
            "max_price": None,
            "spice_preference": None,
            "doc_type_filter": None,
        })
        results = hybrid_search(query_vectors, filters=allergen_only_filter)

    # Step 4: Format for LLM
    context_str, sources = format_context(results)
    return context_str, sources


# ── Run standalone ─────────────────────────────────────────────
if __name__ == "__main__":
    from query_intent import classify_intent

    test_queries = [
        "What vegetarian starters do you have under ₹250?",
        "I'm allergic to dairy. What can I safely eat?",
        "What time does the restaurant close on weekends?",
        "How do I cancel my table booking?",
    ]

    for q in test_queries:
        print(f"\n{'='*55}")
        print(f"Query: {q}")
        intent = classify_intent(q)
        print(f"Intent: {intent['intent_type']} | "
              f"veg={intent['veg_only']} | "
              f"allergens={intent['exclude_allergens']} | "
              f"doc_type={intent['doc_type_filter']}")

        context, sources = retrieve(q, intent)
        print(f"\nTop sources:")
        for s in sources:
            name = s.get('item_name') or s.get('source_doc')
            print(f"  - {name} | {s.get('content_type')} | score={s.get('score', 0):.3f}")
        print(f"\nContext preview (first 300 chars):")
        print(context[:300])
