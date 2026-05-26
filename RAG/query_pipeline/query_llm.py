"""
query_pipeline/query_llm.py
─────────────────────────────
WHAT:  Sends retrieved context + conversation history to Cerebras
       and streams back a customer-facing response.

WHY CEREBRAS:
  Cerebras CS-3 wafer-scale chips run Llama-3.3-70B at ~1,800 tok/s —
  roughly 10-20x faster than standard GPU inference (OpenAI, Anthropic).
  For a restaurant chatbot where customers expect instant responses,
  this latency difference matters enormously.

  Comparison:
    Cerebras Llama-3.3-70B : ~1,800 tok/s  → first token in ~100ms
    OpenAI GPT-4o           : ~80 tok/s    → first token in ~500ms
    Groq Llama-3.3-70B      : ~280 tok/s   → first token in ~200ms

  For a typical restaurant response (150 tokens), Cerebras takes ~80ms
  vs ~1,800ms for GPT-4o. At a real restaurant, that's the difference
  between a chatbot that feels live vs one that feels like it's thinking.

  Cerebras API is OpenAI-compatible — just swap the base URL.

PROMPT STRATEGY:
  We use a layered system prompt approach:
    1. Restaurant persona + constraints (who you are)
    2. Safety rules for allergens (non-negotiable)
    3. Retrieved context (what you know for THIS query)
    4. Conversation history (what was said before)
    5. Customer query (what they want NOW)

  This order matters: safety rules come before context so the LLM
  cannot be "context-jailbroken" by retrieved documents.

STREAMING:
  Cerebras supports SSE streaming. We yield chunks so the calling
  layer (API/websocket) can stream to the frontend in real-time.
"""

import os
import sys
import json
import requests
from typing import Generator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from query_config import (
    CEREBRAS_API_KEY, CEREBRAS_API_URL, CEREBRAS_MODEL, MAX_HISTORY_TURNS
)


# ── System prompt ───────────────────────────────────────────────
SYSTEM_PROMPT = """You are the AI assistant for Restaurant Grand Chennai, a premium Indian restaurant.
You help customers with menu recommendations, table bookings, allergen information, opening hours, and restaurant policies.

## Your Persona
- Warm, professional, and knowledgeable about Indian cuisine
- Specific and helpful — never vague ("I think" → never; "The menu has" → always)
- Concise: answer in 2-4 sentences unless a list is needed

## CRITICAL SAFETY RULES — NEVER VIOLATE
1. ALLERGEN SAFETY: If a customer mentions an allergy or intolerance, ONLY recommend dishes
   that are confirmed allergen-free from the retrieved context. If unsure, say so explicitly
   and recommend they inform the waiter on arrival for confirmation.
2. NEVER INVENT: Only state prices, allergens, and dish names from the context provided.
   If the context doesn't contain the answer, say "I don't have that information — please
   call us or ask the waiter on arrival."
3. BOOKING ESCALATION: For actual reservations, direct customers to call (+91-44-XXXX-XXXX)
   or use the online booking form — you cannot book on their behalf.

## Response Format
- For menu queries: mention dish name, price (₹), veg/non-veg, and allergens if relevant
- For booking queries: explain the policy clearly, then provide next steps
- For opening hours: be precise about the day/time
- Do NOT use markdown headers. Use plain conversational text.
- Lists are OK for multiple dishes (max 4 items)
"""


# ── Prompt templates per intent type ───────────────────────────
INTENT_CONTEXT_HEADERS = {
    "menu":     "MENU INFORMATION FROM OUR KNOWLEDGE BASE:",
    "booking":  "BOOKING POLICY FROM OUR KNOWLEDGE BASE:",
    "info":     "RESTAURANT INFORMATION FROM OUR KNOWLEDGE BASE:",
    "policy":   "POLICY INFORMATION FROM OUR KNOWLEDGE BASE:",
    "general":  "RELEVANT INFORMATION FROM OUR KNOWLEDGE BASE:",
}


def build_messages(
    query:         str,
    context:       str,
    intent:        dict,
    history:       list,
    is_no_results: bool = False,
) -> list:
    """
    Constructs the messages array for the Cerebras API call.

    Structure:
      messages = [
          {"role": "system",    "content": SYSTEM_PROMPT},
          {"role": "user",      "content": "<history turn 1>"},
          {"role": "assistant", "content": "<history turn 1 reply>"},
          ...  (up to MAX_HISTORY_TURNS)
          {"role": "user",      "content": "<current query with context>"},
      ]

    WHY context is injected into the user message (not system):
    The system prompt is static and cached. Injecting dynamic context
    into the user message avoids cache-busting the system prompt,
    which improves latency on subsequent turns.
    """
    intent_type = intent.get("intent_type", "general")
    context_header = INTENT_CONTEXT_HEADERS.get(intent_type, INTENT_CONTEXT_HEADERS["general"])

    # Build allergen warning if relevant
    allergen_warning = ""
    if intent.get("exclude_allergens"):
        allergens_str = ", ".join(intent["exclude_allergens"])
        allergen_warning = (
            f"\n⚠️  ALLERGEN ALERT: This customer has indicated sensitivity to: {allergens_str.upper()}. "
            f"Only recommend dishes that are confirmed free of these allergens in the context below. "
            f"If no safe options are found, say so explicitly.\n"
        )

    # Build user message with injected context
    if is_no_results or not context.strip():
        context_block = (
            "NO RELEVANT INFORMATION FOUND in the knowledge base for this query. "
            "Answer from general restaurant knowledge if appropriate, or let the customer "
            "know you'll need to connect them with staff for this specific query."
        )
    else:
        context_block = f"{context_header}\n\n{context}"

    user_message = f"{allergen_warning}{context_block}\n\n---\n\nCUSTOMER QUERY: {query}"

    # Build messages array
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add conversation history (last N turns)
    recent_history = history[-(MAX_HISTORY_TURNS * 2):]  # *2 for user+assistant pairs
    messages.extend(recent_history)

    # Add current query
    messages.append({"role": "user", "content": user_message})

    return messages


# ── Cerebras API call (streaming) ───────────────────────────────
def call_cerebras_stream(messages: list) -> Generator[str, None, None]:
    """
    Calls the Cerebras API with streaming enabled.
    Yields text chunks as they arrive (SSE stream parsing).

    WHY streaming vs. blocking call:
    Blocking: customer sees nothing until full response ready (~5-10s for long replies)
    Streaming: customer sees first word in ~100ms, response feels instant

    The Cerebras API is OpenAI-compatible, so we use the same
    SSE format: data: {"choices": [{"delta": {"content": "..."}}]}
    """
    if not CEREBRAS_API_KEY:
        yield "⚠️  CEREBRAS_API_KEY not set. Add it to your .env file."
        return

    headers = {
        "Authorization": f"Bearer {CEREBRAS_API_KEY}",
        "Content-Type":  "application/json",
    }

    payload = {
        "model":       CEREBRAS_MODEL,
        "messages":    messages,
        "max_tokens":  512,
        "temperature": 0.3,    # low temp for factual restaurant queries
        "stream":      True,
    }

    try:
        with requests.post(
            CEREBRAS_API_URL,
            headers=headers,
            json=payload,
            stream=True,
            timeout=30,
        ) as response:
            if response.status_code != 200:
                yield f"❌ Cerebras API error {response.status_code}: {response.text[:200]}"
                return

            for line in response.iter_lines():
                if not line:
                    continue
                line_str = line.decode("utf-8")
                if line_str.startswith("data: "):
                    data_str = line_str[6:]
                    if data_str == "[DONE]":
                        return
                    try:
                        data    = json.loads(data_str)
                        delta   = data["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue

    except requests.exceptions.Timeout:
        yield "\n\n[Response timed out. Please try again.]"
    except requests.exceptions.ConnectionError:
        yield "\n\n[Cannot reach Cerebras API. Check your connection.]"
    except Exception as e:
        yield f"\n\n[Unexpected error: {e}]"


def call_cerebras_blocking(messages: list) -> str:
    """
    Non-streaming version — collects the full response and returns it.
    Useful for CLI testing, batch processing, or when streaming isn't needed.
    """
    full_response = ""
    for chunk in call_cerebras_stream(messages):
        full_response += chunk
    return full_response


# ── Master LLM call ─────────────────────────────────────────────
def generate_response(
    query:    str,
    context:  str,
    intent:   dict,
    history:  list,
    stream:   bool = False,
) -> str | Generator[str, None, None]:
    """
    Entry point for the LLM layer.
    Builds messages and calls Cerebras.

    stream=True  → returns a generator (for websocket/SSE APIs)
    stream=False → returns full string (for CLI / batch)
    """
    is_no_results = not context.strip()
    messages      = build_messages(query, context, intent, history, is_no_results)

    if stream:
        return call_cerebras_stream(messages)
    else:
        return call_cerebras_blocking(messages)


# ── Run standalone ─────────────────────────────────────────────
if __name__ == "__main__":
    # Quick smoke test — no Qdrant needed, tests prompt construction only
    test_context = """[Source 1: Complete Menu.docx]
Item: Gobi Manchurian
Price: ₹220
Allergens: soy
Type: vegetarian
Spice: spicy

Gobi Manchurian
Category: Starter
Type: Vegetarian
Price: ₹220
Description: Crispy cauliflower florets tossed in spicy Indo-Chinese sauce.
Contains Allergens: Soy
Spice Level: Spicy"""

    test_intent = {
        "intent_type":       "menu",
        "veg_only":          True,
        "exclude_allergens": [],
        "max_price":         300,
        "spice_preference":  "spicy",
        "item_name":         None,
        "is_allergen_query": False,
    }

    print("Testing Cerebras LLM call (blocking)...")
    response = generate_response(
        query   = "What spicy veg starters do you have under ₹300?",
        context = test_context,
        intent  = test_intent,
        history = [],
        stream  = False,
    )
    print(f"\nResponse:\n{response}")
