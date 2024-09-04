import pytest
from retriever.scripts.storage.storage_manager import StorageManager as RetrieverStorageManager
from knowledge_extractor.scripts.storage.storage_manager import StorageManager
from retriever.scripts.initialiser import initialise_embed_model
from retriever.scripts.config import get_env_vars, get_neo4j_config, get_qdrant_config
from llama_index.core.schema import TextNode

@pytest.fixture
def storage_manager():
    # Load environment variables
    env_vars = get_env_vars()

    # Neo4j and Qdrant configurations
    neo4j_config = get_neo4j_config(env_vars)
    qdrant_config = get_qdrant_config(env_vars)

    # Initialise storage manager
    return StorageManager(neo4j_config, qdrant_config), RetrieverStorageManager(neo4j_config, qdrant_config)


def test_storage_and_retrieval_integration(storage_manager, retriever_storage_manager):
    # Create a sample node to store
    sample_node = TextNode(
        text="This is a test node.",
        metadata={"title": "Test Node", "type": "text"}
    )

    # Store node in Neo4j and Qdrant
    storage_manager.store_nodes([sample_node])

    # Retrieve the node from Neo4j
    retrieved_node = retriever_storage_manager.neo4j_client.get_node_by_llama_id(sample_node.node_id)

    # Check if the retrieved node matches the stored node
    assert retrieved_node.text == sample_node.text
    assert retrieved_node.metadata["title"] == "Test Node"

    # Clean up
    storage_manager.close()
    retriever_storage_manager.close()
