from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import (
    CondenseQuestionChatEngine,
)

from ingestion.indexer import build_index
from query.query_pipeline import build_query_engine


def main():

    #print("Building index...")

    #index = build_index("./docs")

    query_engine = build_query_engine(index)

    memory = ChatMemoryBuffer.from_defaults(
        token_limit=4000,
    )

    chat_engine = CondenseQuestionChatEngine.from_defaults(
        query_engine=query_engine,
        memory=memory,
        verbose=True,
    )

    print("\nRestaurant assistant ready.\n")

    while True:

        question = input("Customer: ").strip()

        if question.lower() in [
            "quit",
            "exit",
        ]:
            break

        response = chat_engine.chat(question)

        print(f"\nAssistant: {response}\n")


if __name__ == "__main__":
    main()