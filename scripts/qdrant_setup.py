import qdrant_client
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext


def setup_qdrant_client(host, port, collection_name):
    client = qdrant_client.QdrantClient(
        host=host,
        port=port,
    )

    vector_store = QdrantVectorStore(
        client=qdrant_client, collection_name=collection_name
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    return client, vector_store, storage_context
