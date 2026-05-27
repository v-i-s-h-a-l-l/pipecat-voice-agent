from qdrant_client import QdrantClient

from llama_index.core import (
    StorageContext,
    VectorStoreIndex,
)

from llama_index.vector_stores.qdrant import (
    QdrantVectorStore,
)

from llama_index.core.retrievers import (
    VectorIndexRetriever,
)

from llama_index.core.query_engine import (
    RetrieverQueryEngine,
)

from llama_index.core.response_synthesizers import (
    get_response_synthesizer,
)

from llama_index.core.postprocessor import (
    SimilarityPostprocessor,
)

from config import (
    COLLECTION_NAME,
    EMBED_MODEL,
    QDRANT_PATH,
    RESPONSE_LLM,
    TOP_K,
)


# =========================================================
# BUILD QUERY ENGINE
# =========================================================

def build_query_engine():

    # =====================================================
    # LOAD EXISTING QDRANT STORAGE
    # =====================================================

    client = QdrantClient(
        path=QDRANT_PATH
    )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
    )

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
    )

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
        embed_model=EMBED_MODEL,
    )

    # =====================================================
    # RETRIEVER
    # =====================================================

    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=TOP_K,
    )

    # =====================================================
    # RESPONSE SYNTHESIZER
    # =====================================================

    synthesizer = get_response_synthesizer(
        llm=RESPONSE_LLM,
        response_mode="compact",
    )

    # =====================================================
    # QUERY ENGINE
    # =====================================================

    return RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=synthesizer,
        node_postprocessors=[
            SimilarityPostprocessor(
                similarity_cutoff=0.3
            )
        ],
    )