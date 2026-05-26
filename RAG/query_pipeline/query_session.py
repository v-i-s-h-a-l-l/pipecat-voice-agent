"""
query_pipeline/query_session.py
────────────────────────────────
WHAT:  Manages multi-turn conversation state.

WHY SESSION STATE MATTERS:
  Without session state, every query is independent:
    Turn 1: "I'm allergic to dairy" → filter applied, good response
    Turn 2: "What starters do you have?" → filter FORGOTTEN, dairy dishes returned
    → Customer with dairy allergy now sees dangerous recommendations

  With session state:
    Allergen declarations persist for the entire session.
    Veg preferences persist.
    Conversation history is maintained for coherent multi-turn dialogue.

  This is especially important for allergen safety — a customer who
  mentions an allergy once should never need to repeat it.

SESSION STRUCTURE:
    {
        session_id:     str,
        history:        list,   ← [{role, content}] for LLM context
        context:        dict,   ← persistent user preferences
            {
                allergens: list,
                is_veg:    bool,
                name:      str | None,   ← if customer introduces themselves
            }
        turn_count:     int,
        created_at:     float,
        last_active:    float,
    }
"""

import uuid
import time
from typing import Optional


# In-memory session store
# Production: replace with Redis (TTL support, multi-process safe)
_sessions: dict[str, dict] = {}

SESSION_TTL_SECONDS = 3600   # 1 hour inactivity → session expires


def create_session() -> str:
    """Create a new session and return its ID."""
    session_id = str(uuid.uuid4())
    _sessions[session_id] = {
        "session_id":  session_id,
        "history":     [],
        "context":     {
            "allergens": [],
            "is_veg":    False,
            "name":      None,
        },
        "turn_count":  0,
        "created_at":  time.time(),
        "last_active": time.time(),
    }
    return session_id


def get_session(session_id: str) -> dict | None:
    """Get session, checking TTL. Returns None if expired or missing."""
    session = _sessions.get(session_id)
    if not session:
        return None
    if time.time() - session["last_active"] > SESSION_TTL_SECONDS:
        del _sessions[session_id]
        return None
    return session


def get_or_create_session(session_id: str | None) -> tuple[str, dict]:
    """Get existing session or create a new one. Returns (id, session)."""
    if session_id:
        session = get_session(session_id)
        if session:
            return session_id, session
    new_id = create_session()
    return new_id, _sessions[new_id]


def update_session_from_intent(session: dict, intent: dict) -> None:
    """
    Updates persistent session context from the current intent.

    WHY allergens accumulate:
    If the customer says "I'm allergic to dairy" in turn 1 and
    "I'm also sensitive to gluten" in turn 3, both must be tracked.
    We union (not replace) the allergen list.

    WHY veg preference is sticky:
    "I'm vegetarian" → should apply to all subsequent queries.
    But "Show me non-veg options" is an override (handled separately).
    """
    ctx = session["context"]

    # Accumulate allergens (union, never overwrite)
    new_allergens = intent.get("exclude_allergens", [])
    if new_allergens:
        existing = set(ctx.get("allergens", []))
        ctx["allergens"] = list(existing.union(set(new_allergens)))

    # Sticky veg preference
    if intent.get("veg_only"):
        ctx["is_veg"] = True


def add_turn(session: dict, query: str, response: str) -> None:
    """
    Adds a completed turn to the session history.
    Trims history to MAX_HISTORY_TURNS to control context window.
    """
    from query_config import MAX_HISTORY_TURNS

    session["history"].append({"role": "user",      "content": query})
    session["history"].append({"role": "assistant",  "content": response})
    session["turn_count"]  += 1
    session["last_active"]  = time.time()

    # Keep only the last N turns (user+assistant = 2 messages per turn)
    max_messages = MAX_HISTORY_TURNS * 2
    if len(session["history"]) > max_messages:
        session["history"] = session["history"][-max_messages:]


def get_session_context(session: dict) -> dict:
    """Returns the persistent context to pass to intent classifier."""
    return session.get("context", {})


def list_active_sessions() -> list:
    """Returns list of active (non-expired) session IDs."""
    now = time.time()
    active = [
        sid for sid, s in _sessions.items()
        if now - s["last_active"] <= SESSION_TTL_SECONDS
    ]
    return active


def clear_expired_sessions() -> int:
    """Removes expired sessions. Call periodically."""
    now     = time.time()
    expired = [
        sid for sid, s in _sessions.items()
        if now - s["last_active"] > SESSION_TTL_SECONDS
    ]
    for sid in expired:
        del _sessions[sid]
    return len(expired)
