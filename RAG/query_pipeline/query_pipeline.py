"""
query_pipeline/query_pipeline.py
──────────────────────────────────
WHAT:  Master orchestrator for the query pipeline.
       Ties together intent → retrieval → LLM into one callable.

QUERY PIPELINE STAGES:
  1. Session management    → persistent state across turns
  2. Intent classification → what does the customer want?
  3. Retrieval             → fetch relevant chunks from Qdrant
  4. Response generation   → Cerebras LLM generates the answer
  5. History update        → save turn to session

USAGE:
  # Interactive CLI
  python query_pipeline.py

  # Single query (non-interactive)
  python query_pipeline.py --query "What veg dishes do you have?"

  # Test a set of predefined queries
  python query_pipeline.py --test

  # Disable streaming (print full response at once)
  python query_pipeline.py --no-stream
"""

import sys
import os
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from query_intent    import classify_intent
from query_retriever import retrieve
from query_llm       import generate_response
from query_session   import get_or_create_session, update_session_from_intent, add_turn, get_session_context


# ── Master query function ───────────────────────────────────────
def query(
    user_input:  str,
    session_id:  str | None = None,
    stream:      bool = True,
    verbose:     bool = False,
) -> dict:
    """
    Full query pipeline: input → structured response.

    Args:
        user_input: raw customer message
        session_id: existing session ID (or None to start new session)
        stream:     True → generator; False → full string
        verbose:    print debug info (intent, retrieved sources)

    Returns:
        {
            session_id:  str,
            response:    str | generator,
            sources:     list,
            intent:      dict,
            latency_ms:  dict,   ← per-stage timing
        }
    """
    timings = {}
    t_total = time.time()

    # ── Stage 1: Session ─────────────────────────────────────
    t = time.time()
    session_id, session = get_or_create_session(session_id)
    session_ctx         = get_session_context(session)
    timings["session_ms"] = int((time.time() - t) * 1000)

    # ── Stage 2: Intent classification ───────────────────────
    t = time.time()
    intent = classify_intent(user_input, session_context=session_ctx)
    update_session_from_intent(session, intent)
    timings["intent_ms"] = int((time.time() - t) * 1000)

    if verbose:
        print(f"\n[Intent]")
        print(f"  type            : {intent['intent_type']}")
        print(f"  veg_only        : {intent['veg_only']}")
        print(f"  exclude_allergens: {intent['exclude_allergens']}")
        print(f"  max_price       : {intent['max_price']}")
        print(f"  spice_pref      : {intent['spice_preference']}")
        print(f"  doc_type_filter : {intent['doc_type_filter']}")
        print(f"  item_name       : {intent['item_name']}")

    # ── Stage 3: Retrieval ────────────────────────────────────
    t = time.time()
    context_str, sources = retrieve(user_input, intent)
    timings["retrieval_ms"] = int((time.time() - t) * 1000)

    if verbose:
        print(f"\n[Retrieval] {len(sources)} sources found in {timings['retrieval_ms']}ms")
        for s in sources:
            name = s.get("item_name") or s.get("source_doc")
            print(f"  - {name} | {s.get('content_type')} | score={s.get('score', 0):.3f}")

    # ── Stage 4: LLM response generation ─────────────────────
    t = time.time()
    history  = session.get("history", [])
    response = generate_response(
        query   = user_input,
        context = context_str,
        intent  = intent,
        history = history,
        stream  = stream,
    )
    timings["llm_start_ms"] = int((time.time() - t) * 1000)
    timings["total_ms"]     = int((time.time() - t_total) * 1000)

    # ── Stage 5: History update (after response consumed) ────
    # Note: for streaming, history is updated by the caller
    # after the generator is exhausted. For blocking, update now.
    if not stream:
        add_turn(session, user_input, response)

    return {
        "session_id":  session_id,
        "response":    response,
        "sources":     sources,
        "intent":      intent,
        "latency_ms":  timings,
    }


# ── Interactive CLI ─────────────────────────────────────────────
def run_interactive(stream=True, verbose=False):
    """
    Multi-turn interactive CLI for testing the query pipeline.
    Maintains session across turns.
    """
    print("\n" + "═" * 60)
    print("  🍽️  Restaurant Grand Chennai — AI Assistant")
    print("  Powered by Cerebras + Qdrant Hybrid Search")
    print("═" * 60)
    print("  Type your question. Type 'quit' or 'exit' to stop.")
    print("  Type 'sources' to show last retrieved sources.")
    print("  Type 'debug' to toggle verbose mode.\n")

    session_id  = None
    last_sources = []
    debug_mode   = verbose

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye! 🙏")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            print("Goodbye! 🙏")
            break

        if user_input.lower() == "sources":
            if last_sources:
                print("\n[Last retrieved sources]")
                for s in last_sources:
                    name = s.get("item_name") or s.get("source_doc")
                    print(f"  - {name} | {s.get('content_type')} | score={s.get('score',0):.3f}")
            else:
                print("[No sources retrieved yet]")
            continue

        if user_input.lower() == "debug":
            debug_mode = not debug_mode
            print(f"[Debug mode: {'ON' if debug_mode else 'OFF'}]")
            continue

        result = query(
            user_input = user_input,
            session_id = session_id,
            stream     = stream,
            verbose    = debug_mode,
        )

        session_id   = result["session_id"]
        last_sources = result["sources"]
        response     = result["response"]

        print(f"\nAssistant: ", end="", flush=True)

        if stream:
            # Stream response to terminal
            full_response = ""
            for chunk in response:
                print(chunk, end="", flush=True)
                full_response += chunk
            print()  # newline after streaming
            # Update history after streaming is done
            from query_session import get_session, add_turn
            session = get_session(session_id)
            if session:
                add_turn(session, user_input, full_response)
        else:
            print(response)

        if debug_mode:
            t = result["latency_ms"]
            print(f"\n[Latency] intent={t.get('intent_ms')}ms | "
                  f"retrieval={t.get('retrieval_ms')}ms | "
                  f"llm_start={t.get('llm_start_ms')}ms")

        print()


# ── Test suite ──────────────────────────────────────────────────
def run_test_suite():
    """Runs a predefined set of queries to verify the full pipeline."""

    test_cases = [
        # (query, expected_intent_type)
        ("What vegetarian starters do you have?",                   "menu"),
        ("I'm allergic to dairy. What's safe for me to eat?",       "menu"),
        ("How much does Gobi Manchurian cost?",                     "menu"),
        ("Can I book a table for this Saturday at 7pm for 2?",      "booking"),
        ("What's your cancellation policy?",                        "booking"),
        ("What time do you open on Sunday?",                        "info"),
        ("Recommend something spicy and non-veg under ₹400",        "menu"),
        ("Do you have parking?",                                    "info"),
    ]

    print("\n" + "═" * 60)
    print("  🧪  Query Pipeline Test Suite")
    print("═" * 60)

    session_id = None
    passed     = 0

    for i, (query_text, expected_intent) in enumerate(test_cases):
        print(f"\n[Test {i+1}/{len(test_cases)}]")
        print(f"  Query  : {query_text}")

        result = query(
            user_input = query_text,
            session_id = session_id,
            stream     = False,
            verbose    = False,
        )

        session_id  = result["session_id"]
        got_intent  = result["intent"]["intent_type"]
        response    = result["response"]
        t           = result["latency_ms"]
        n_sources   = len(result["sources"])

        intent_ok = got_intent == expected_intent
        has_response = len(response) > 20

        status = "✅" if (intent_ok and has_response) else "⚠️ "
        if intent_ok and has_response:
            passed += 1

        print(f"  Intent : {got_intent} (expected: {expected_intent}) {status}")
        print(f"  Sources: {n_sources} retrieved")
        print(f"  Latency: retrieval={t.get('retrieval_ms')}ms | "
              f"total={t.get('total_ms')}ms")
        print(f"  Response preview: {response[:120]}...")

    print(f"\n{'═' * 60}")
    print(f"  Results: {passed}/{len(test_cases)} tests passed")
    print(f"{'═' * 60}")


# ── Argument parser ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Restaurant Grand Chennai — Query Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python query_pipeline.py                          # interactive CLI
  python query_pipeline.py --query "what's open?"  # single query
  python query_pipeline.py --test                  # run test suite
  python query_pipeline.py --no-stream --verbose   # debug mode
        """
    )
    parser.add_argument("--query",     type=str,  default=None,
                        help="Run a single query and exit")
    parser.add_argument("--test",      action="store_true",
                        help="Run the predefined test suite")
    parser.add_argument("--no-stream", action="store_true",
                        help="Disable streaming (print full response at once)")
    parser.add_argument("--verbose",   action="store_true",
                        help="Print intent + retrieved sources for each query")
    return parser.parse_args()


# ── Entrypoint ──────────────────────────────────────────────────
if __name__ == "__main__":
    args    = parse_args()
    do_stream = not args.no_stream

    if args.test:
        run_test_suite()

    elif args.query:
        result = query(
            user_input = args.query,
            session_id = None,
            stream     = do_stream,
            verbose    = args.verbose,
        )
        print(f"\nAssistant: ", end="", flush=True)
        if do_stream:
            for chunk in result["response"]:
                print(chunk, end="", flush=True)
            print()
        else:
            print(result["response"])

        if args.verbose:
            t = result["latency_ms"]
            print(f"\n[Latency] {t}")
            print(f"[Sources] {len(result['sources'])} retrieved")

    else:
        run_interactive(stream=do_stream, verbose=args.verbose)
