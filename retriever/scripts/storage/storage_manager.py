import logging
from scripts.storage.graph_db_setup import Neo4jClient
from scripts.storage.qdrant_setup import setup_qdrant_client
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
import time


class StorageManager:
    def __init__(self, neo4j_config, qdrant_config, max_retries=10, wait_time=5):
        self.neo4j_client = Neo4jClient(**neo4j_config)

        self.embed_model = Settings.embed_model

        for attempt in range(max_retries):
            try:
                logging.info(f"Attempting to set up Qdrant client (Attempt {attempt + 1}/{max_retries})")
                
                (self.qdrant_client,
                self.text_vector_store,
                self.image_vector_store,
                self.text_storage_context,
                self.image_storage_context,
                ) = setup_qdrant_client(**qdrant_config)
                break
            except Exception as e:
                logging.error(f"Error setting up Qdrant client: {e}")
                if attempt < max_retries - 1:
                    logging.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logging.error("Max retries reached. Could not set up Qdrant client.")
                    raise Exception("Failed to set up Qdrant client after several attempts.")




    def build_index(self):
        logging.info("Building VectorStoreIndex from Qdrant vector store.")
        index = VectorStoreIndex.from_vector_store(
            self.text_vector_store, embed_model=self.embed_model
        )
        logging.info("Index built successfully.")
        return index

    def vector_search(self, query_vector, top_k, node_type=None):

        filter_condition = (
            {"must": [{"key": "type", "match": {"value": node_type}}]}
            if node_type
            else None
        )
        return self.qdrant_client.search(
            collection_name=self.text_vector_store.collection_name,
            query_vector=query_vector,
            with_payload=True,
            limit=top_k,
            query_filter=filter_condition,
        )

    def close(self):
        self.neo4j_client.close()
