from llama_index.core.memory import (
    ChatMemoryBuffer,
)

from llama_index.core.chat_engine import (
    CondenseQuestionChatEngine,
)

from query.query_pipeline import (
    build_query_engine,
)


# =========================================================
# STATIC ROUTING
# =========================================================

GREETINGS = [
    "hi",
    "hello",
    "hey",
    "good morning",
    "good evening",
]

THANKS = [
    "thanks",
    "thank you",
]

BYES = [
    "bye",
    "goodbye",
    "see you",
]


def handle_static_query(query):

    q = query.lower().strip()

    # =====================================================
    # GREETINGS
    # =====================================================

    if q in GREETINGS:

        return (
            "Hello! Welcome to Restaurant Grand Chennai. "
            "How can I assist you today? "
            "I can help with menu recommendations, "
            "table bookings, allergens, and restaurant information."
        )

    # =====================================================
    # THANKS
    # =====================================================

    if q in THANKS:

        return (
            "You're welcome! "
            "Please let me know if you need anything else."
        )

    # =====================================================
    # GOODBYE
    # =====================================================

    if q in BYES:

        return (
            "Thank you for visiting Restaurant Grand Chennai. "
            "Have a wonderful day!"
        )

    return None


# =========================================================
# MAIN
# =========================================================

def main():

    print("Loading restaurant assistant...")

    query_engine = build_query_engine()

    # =====================================================
    # MEMORY
    # =====================================================

    memory = ChatMemoryBuffer.from_defaults(
        token_limit=4000,
    )

    # =====================================================
    # CHAT ENGINE
    # =====================================================

    chat_engine = CondenseQuestionChatEngine.from_defaults(
        query_engine=query_engine,
        memory=memory,
        verbose=False,
    )

    print("\nRestaurant assistant ready.\n")

    print("Commands:")
    print("  /clear  -> clear conversation memory")
    print("  /exit   -> exit assistant\n")

    while True:

        question = input("Customer: ").strip()

        # =================================================
        # EXIT
        # =================================================

        if question.lower() in [
            "/exit",
            "exit",
            "quit",
        ]:
            print("\nAssistant: Goodbye!\n")
            break

        # =================================================
        # CLEAR MEMORY
        # =================================================

        if question.lower() == "/clear":

            memory.reset()

            print(
                "\nAssistant: Conversation memory cleared.\n"
            )

            continue

        # =================================================
        # STATIC RESPONSES
        # =================================================

        static_response = handle_static_query(
            question
        )

        if static_response:

            print(f"\nAssistant: {static_response}\n")

            continue

        # =================================================
        # CHAT ENGINE
        # =================================================

        try:

            response = chat_engine.chat(
                question
            )

            print(f"\nAssistant: {response}\n")

        except Exception as e:

            print(
                "\nAssistant: "
                "Sorry, something went wrong.\n"
            )

            print(e)


if __name__ == "__main__":
    main()