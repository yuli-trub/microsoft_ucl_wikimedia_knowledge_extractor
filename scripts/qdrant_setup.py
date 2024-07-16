from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext
from qdrant_client.http.models import VectorParams, Distance
import logging

logger = logging.getLogger(__name__)

# was trying to fix a bug and wrote all that, but apparently it was env var typo... but this works


def setup_qdrant_client(host, port, collection_name):

    try:
        client = QdrantClient(
            host=host,
            port=port,
        )
        logger.info("Connected to Qdrant client successfully.")
    except Exception as e:
        logger.error(f"Error connecting to Qdrant client: {e}")
        raise

    try:
        collection_exists = False
        try:
            client.get_collection(collection_name=collection_name)
            collection_exists = True
            logger.info(f"Collection '{collection_name}' already exists.")
        except Exception as e:
            logger.info(
                f"Collection '{collection_name}' does not exist. Creating new collection."
            )

        if not collection_exists:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
            )
            logger.info(f"Collection '{collection_name}' created successfully.")
    except Exception as e:
        logger.error(f"Error checking or creating collection: {e}")
        raise

    try:
        vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        logger.info("Connected to Qdrant vector store successfully.")
    except Exception as e:
        logger.error(f"Error connecting to Qdrant vector store: {e}")
        raise

    return client, vector_store, storage_context


def verify_qdrant(client, collection_name):
    collections = client.get_collections()
    logger.info(
        f"Collections in Qdrant: {[collection.name for collection in collections.collections]}"
    )
    response = client.scroll(
        collection_name=collection_name,
        limit=10,
    )
    logger.info(f"Vectors in collection: {len(response.points)}")
