"""
query_pipeline/query_intent.py
───────────────────────────────
WHAT:  Classifies the customer query into a structured intent object
       that drives routing, filter-building, and prompt selection.

WHY INTENT CLASSIFICATION:
  Without understanding intent, every query gets the same treatment.
  With intent, we can:
    - "Do you have veg dishes?"      → veg_only=True filter
    - "I'm allergic to gluten"       → exclude_allergens=["gluten"] (SAFETY)
    - "Book a table for Saturday"    → route to booking docs only
    - "What time do you open?"       → route to restaurant_info only
    - "What's the price of X?"       → route to menu + exact item match
    - "Cancel my reservation"        → route to booking_system

  Intent extraction is regex-first (fast, deterministic) then LLM
  for complex multi-intent queries.

OUTPUT: Intent dict:
    {
        intent_type:       str,   ← "menu" | "booking" | "policy" | "info" | "general"
        veg_only:          bool,
        exclude_allergens: list,
        max_price:         int | None,
        spice_preference:  str | None,
        item_name:         str | None,
        doc_type_filter:   str | None,
        is_allergen_query: bool,   ← triggers safety mode
        raw_query:         str,
    }
"""

import re


# ── Allergen keyword map ────────────────────────────────────────
ALLERGEN_KEYWORDS = {
    "dairy":   ["dairy", "milk", "lactose", "paneer", "cream", "butter", "cheese", "yogurt", "curd"],
    "gluten":  ["gluten", "wheat", "flour", "bread", "naan", "roti", "maida"],
    "nuts":    ["nuts", "peanut", "almond", "cashew", "walnut", "pistachio", "groundnut"],
    "soy":     ["soy", "soya", "tofu"],
    "egg":     ["egg", "eggs"],
    "shellfish":["shellfish", "prawn", "shrimp", "crab", "lobster", "oyster"],
}

# ── Spice level map ─────────────────────────────────────────────
SPICE_KEYWORDS = {
    "mild":       ["mild", "not spicy", "less spicy", "no spice", "bland"],
    "medium":     ["medium", "moderate", "medium spice"],
    "spicy":      ["spicy", "hot", "fiery"],
    "extra_spicy":["extra spicy", "very spicy", "extremely spicy"],
}

# ── Veg intent signals ──────────────────────────────────────────
VEG_SIGNALS     = ["vegetarian", "veg only", "no meat", "no non-veg",
                   "plant based", "i don't eat meat", "i am vegetarian",
                   "we are vegetarian"]
NON_VEG_SIGNALS = ["non-veg", "non veg", "chicken", "mutton", "prawn",
                   "fish", "seafood", "lamb", "egg", "meat"]

# ── Doc-type routing signals ────────────────────────────────────
BOOKING_SIGNALS  = ["book", "reservation", "reserv", "table", "cancel booking",
                    "cancel reservation", "modify booking", "reschedule"]
INFO_SIGNALS     = ["open", "close", "timing", "hours", "location", "address",
                    "parking", "direction", "how to reach", "where are you",
                    "phone", "contact", "wifi"]
POLICY_SIGNALS   = ["refund", "policy", "complaint", "feedback", "loyalty",
                    "points", "reward", "cancel", "waiting list"]
MENU_SIGNALS     = ["menu", "dish", "food", "eat", "hungry", "starter",
                    "main course", "dessert", "drink", "beverage", "price",
                    "cost", "recommend", "suggest", "allergen", "allergy",
                    "spice", "veg", "non-veg", "biryani", "soup"]


def extract_allergens_from_query(query: str) -> list:
    """
    WHY strict allergen extraction:
    If a customer says "I'm allergic to dairy" we MUST exclude
    dairy-containing dishes. This is a safety-critical extraction,
    not an optional enhancement.
    """
    found = []
    query_lower = query.lower()
    for allergen, keywords in ALLERGEN_KEYWORDS.items():
        if any(kw in query_lower for kw in keywords):
            # Only add if context is allergy/intolerance, not just mention
            allergy_context = any(c in query_lower for c in [
                "allerg", "intolerant", "intolerance", "avoid", "can't eat",
                "cannot eat", "don't want", "without", "no ", "free from",
                "safe for", "i have"
            ])
            if allergy_context:
                found.append(allergen)
    return found


def extract_price_limit(query: str) -> int | None:
    """Extract max price from queries like 'under ₹300' or 'below 250'."""
    patterns = [
        r'under\s*₹?\s*(\d+)',
        r'below\s*₹?\s*(\d+)',
        r'less than\s*₹?\s*(\d+)',
        r'within\s*₹?\s*(\d+)',
        r'budget.*?₹?\s*(\d+)',
        r'₹\s*(\d+)\s+(?:or less|max|maximum)',
    ]
    for p in patterns:
        m = re.search(p, query, re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None


def extract_spice_preference(query: str) -> str | None:
    query_lower = query.lower()
    for level, keywords in SPICE_KEYWORDS.items():
        if any(kw in query_lower for kw in keywords):
            return level
    return None


def extract_item_name(query: str) -> str | None:
    """
    Extract potential dish names from queries like
    'price of Paneer Tikka' or 'tell me about Gobi Manchurian'.
    """
    patterns = [
        r'(?:price|cost|about|details?|info)\s+(?:of|on|for)\s+([A-Z][a-zA-Z\s]+?)(?:\?|$|,|\s+and)',
        r'(?:is|are)\s+([A-Z][a-zA-Z\s]+?)\s+(?:available|good|veg|non)',
    ]
    for p in patterns:
        m = re.search(p, query)
        if m:
            return m.group(1).strip()
    return None


def classify_doc_type(query: str) -> str | None:
    """Route query to the most relevant document type."""
    query_lower = query.lower()

    if any(s in query_lower for s in BOOKING_SIGNALS):
        return "booking_system"
    if any(s in query_lower for s in INFO_SIGNALS):
        return "restaurant_info"
    if any(s in query_lower for s in POLICY_SIGNALS):
        return "customer_experience"
    if any(s in query_lower for s in MENU_SIGNALS):
        return "menu"
    return None   # no specific routing — search all docs


def classify_intent(query: str, session_context: dict = None) -> dict:
    """
    Main intent classifier. Combines regex signals with session context
    (e.g., previously declared allergens carry forward across turns).

    session_context: persistent state across conversation turns
      {
          allergens: list,   ← accumulated from earlier turns
          is_veg:    bool,   ← if declared vegetarian earlier
      }
    """
    sc = session_context or {}
    query_lower = query.lower()

    # ── Allergen extraction (safety-critical) ───────────────────
    turn_allergens = extract_allergens_from_query(query)
    # Merge with session allergens (allergies persist across turns)
    all_allergens  = list(set(sc.get("allergens", []) + turn_allergens))
    is_allergen_query = bool(turn_allergens)

    # ── Veg/non-veg preference ──────────────────────────────────
    veg_only = (
        any(s in query_lower for s in VEG_SIGNALS)
        or sc.get("is_veg", False)
    )

    # ── Price limit ─────────────────────────────────────────────
    max_price = extract_price_limit(query)

    # ── Spice preference ────────────────────────────────────────
    spice_pref = extract_spice_preference(query)

    # ── Document routing ────────────────────────────────────────
    doc_type_filter = classify_doc_type(query)

    # ── Item name ───────────────────────────────────────────────
    item_name = extract_item_name(query)

    # ── Intent type (highest-level label) ───────────────────────
    if doc_type_filter == "booking_system":
        intent_type = "booking"
    elif doc_type_filter == "restaurant_info":
        intent_type = "info"
    elif doc_type_filter == "customer_experience":
        intent_type = "policy"
    elif doc_type_filter == "menu" or any(s in query_lower for s in MENU_SIGNALS):
        intent_type = "menu"
    else:
        intent_type = "general"

    return {
        "intent_type":       intent_type,
        "veg_only":          veg_only,
        "exclude_allergens": all_allergens,
        "max_price":         max_price,
        "spice_preference":  spice_pref,
        "item_name":         item_name,
        "doc_type_filter":   doc_type_filter,
        "is_allergen_query": is_allergen_query,
        "raw_query":         query,
    }


# ── Run standalone ─────────────────────────────────────────────
if __name__ == "__main__":
    test_queries = [
        "Do you have vegetarian starters under ₹200?",
        "I'm allergic to gluten, what can I eat?",
        "I want to book a table for Saturday evening for 4 people",
        "What time does the restaurant open on Sunday?",
        "Tell me about Gobi Manchurian",
        "What is your cancellation policy?",
        "Recommend something spicy and non-veg",
    ]

    for q in test_queries:
        intent = classify_intent(q)
        print(f"\nQuery  : {q}")
        print(f"Intent : {intent['intent_type']}")
        print(f"Veg    : {intent['veg_only']}")
        print(f"Allerg : {intent['exclude_allergens']}")
        print(f"Price  : {intent['max_price']}")
        print(f"DocType: {intent['doc_type_filter']}")
