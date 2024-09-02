from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext
from qdrant_client.http.models import VectorParams, Distance
import logging
from llama_index.core.schema import Node

logger = logging.getLogger(__name__)

def check_collection_exists(client, collection_name, vector_size):
        """Check if collection exists in Qdrant, and create it if it doesn't.
        
        Args:
            collection_name (str): The name of the collection to check or create
            vector_size (int): The size of the vectors to store in the collection

        """
        try:
            collection_exists = False
            try:
                client.get_collection(collection_name=collection_name)
                collection_exists = True
                logger.info(f"Collection '{collection_name}' already exists.")
            except Exception:
                logger.info(
                    f"Collection '{collection_name}' does not exist. Creating new collection."
                )

            if not collection_exists:
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=vector_size, distance=Distance.COSINE
                    ),
                )
                logger.info(f"Collection '{collection_name}' created successfully.")
        except Exception as e:
            logger.error(
                f"Error checking or creating collection '{collection_name}': {e}"
            )
            raise

def setup_qdrant_client(host, port, collection_name):
    """Setup Qdrant client and vector store for storing embeddings.
    
    Args:
        host (str): The host of the Qdrant server
        port (int): The port of the Qdrant server
        collection_name (str): The name of the collection to store embeddings in

    Returns:
        tuple: A tuple containing the Qdrant client, text vector store, image vector store, text storage context, and image storage context
    """
    # Connect to Qdrant client
    try:
        client = QdrantClient(
            host=host,
            port=port,
        )

        logger.info("Connected to Qdrant client successfully.")
    except Exception as e:
        logger.error(f"Error connecting to Qdrant client: {e}")
        raise

    # Check if collections exist and create them if they don't
    text_collection_name = collection_name + "_text"
    image_collection_name = collection_name + "_image"
    check_collection_exists(client, text_collection_name, vector_size=1536)
    check_collection_exists(client, image_collection_name, vector_size=1024)

    # Create vector stores and storage contexts
    try:
        text_vector_store = QdrantVectorStore(
            client=client, collection_name=text_collection_name
        )
        image_vector_store = QdrantVectorStore(
            client=client, collection_name=image_collection_name
        )
        text_storage_context = StorageContext.from_defaults(
            vector_store=text_vector_store
        )
        image_storage_context = StorageContext.from_defaults(
            vector_store=image_vector_store
        )

    except Exception as e:
        logger.error(f"Error connecting to Qdrant vector store: {e}")
        raise

    return (
        client,
        text_vector_store,
        image_vector_store,
        text_storage_context,
        image_storage_context,
    )


def verify_qdrant(client, collection_name):
    """Verify that the Qdrant client is connected and the collection exists."""
    collections = client.get_collections()
    logger.info(
        f"Collections in Qdrant: {[collection.name for collection in collections.collections]}"
    )
    response = client.scroll(
        collection_name=collection_name,
        limit=10,
    )
    logger.info(f"Vectors in collection: {len(response.points)}")
