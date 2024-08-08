from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext
from qdrant_client.http.models import VectorParams, Distance
from qdrant_client import AsyncQdrantClient
import logging
from llama_index.core.schema import Node

logger = logging.getLogger(__name__)


def setup_qdrant_client(host, port, collection_name):

    try:
        client = QdrantClient(
            host=host,
            port=port,
        )
        aclient = AsyncQdrantClient(
            host=host,
            port=port,
        )
        logger.info("Connected to Qdrant client successfully.")
    except Exception as e:
        logger.error(f"Error connecting to Qdrant client: {e}")
        raise

    def check_collection_exists(collection_name, vector_size):
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

    text_collection_name = collection_name + "_text"
    image_collection_name = collection_name + "_image"
    check_collection_exists(text_collection_name, vector_size=1536)

    check_collection_exists(image_collection_name, vector_size=1024)

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
        logger.info("Connected to Qdrant vector stores successfully.")
        # vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
        # storage_context = StorageContext.from_defaults(vector_store=vector_store)
        # logger.info("Connected to Qdrant vector store successfully.")
    except Exception as e:
        logger.error(f"Error connecting to Qdrant vector store: {e}")
        raise

    return (
        client,
        aclient,
        text_vector_store,
        image_vector_store,
        text_storage_context,
        image_storage_context,
    )


def add_node_to_qdrant(vector_store, node, neo_node_id, llama_node_id):
    if node.embedding is not None:
        if node.metadata.get("type") in [
            "image_description",
            "plot_insights",
            "image_entities",
            "plot",
            "image",
        ]:
            node_type = "image"
        else:
            node_type = "text"

        qdrant_node = Node(
            id=llama_node_id,
            embedding=node.embedding,
            metadata={
                "neo4j_node_id": neo_node_id,
                "llama_node_id": llama_node_id,
                "type": node_type,
            },
        )
        vector_store.add([qdrant_node])
        logging.info(f"Node with llama_node_id {llama_node_id} added to Qdrant.")
    else:
        logging.info(
            f"Node with llama_node_id {llama_node_id} has no embedding and was not added to Qdrant."
        )


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
