from llama_index.core.schema import Document, ImageNode, TextNode
import logging
from scripts.storage.graph_db_setup import Neo4jClient
from scripts.storage.qdrant_setup import setup_qdrant_client, add_node_to_qdrant
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

      
    # set up db and check if it exists
    def setup_neo4j_database(neo4j_client: Neo4jClient, database_name: str):
        if not neo4j_client.database_exists(database_name):
            neo4j_client.create_database(database_name)
            neo4j_client.start_database(database_name)
        else:
            logging.info(f"Database {database_name} already exists.")


    def store_nodes_and_relationships(self, nodes):
        neo4j_client = self.neo4j_client
        logging.info(f"Storing: {(len(nodes))} nodes and their relationships in Neo4j")
        node_id_map = {}

        for node in nodes:

            if isinstance(node, Document):
                logging.info(f"Creating document node {node.doc_id}")
                neo_node_id = neo4j_client.create_document_node(node)
                node_id_map[node.doc_id] = neo_node_id
            elif isinstance(node, ImageNode):
                logging.info(f"image node info: {node.metadata['url']}")
                logging.info(
                    f"Creating image node {node.metadata['title']} of type {node.metadata['type']}"
                )
                neo_node_id = neo4j_client.create_image_node(node)
                node_id_map[node.node_id] = neo_node_id
            elif isinstance(node, TextNode) and not isinstance(node, Document):
                logging.info(
                    f"Creating text node {node.metadata['title']} of type {node.metadata['type']}"
                )
                neo_node_id = neo4j_client.create_text_node(node)
                node_id_map[node.node_id] = neo_node_id

        logging.info(f"Nodes created in Neo4j {node_id_map}")

        for node in nodes:
            if node.relationships:

                for relationship, related_node_info in node.relationships.items():

                    relationship_type = relationship.name
                    # logging.info(
                    #     f"Creating relationship {relationship_type} from {node.node_id} to {related_node_info.node_id}"
                    # )
                    to_id = node.node_id
                    from_id = related_node_info.node_id
                    if to_id:
                        check_relationship = neo4j_client.create_relationship(
                            from_id, to_id, relationship_type
                        )
                        # logging.info(f"Relationship created: {check_relationship}")

        logging.info("Nodes and relationships created in Neo4j")

        return node_id_map

    def add_nodes_to_qdrant(self, nodes, id_map):
        logging.info("Adding nodes with embeddings to Qdrant.")
        for node in nodes:
            if node.embedding is not None:
                neo_node_id = id_map.get(node.node_id)
                if neo_node_id:
                    if isinstance(node, ImageNode):
                        add_node_to_qdrant(
                            self.image_vector_store, node, neo_node_id, node.node_id
                        )
                    else:
                        add_node_to_qdrant(
                            self.text_vector_store, node, neo_node_id, node.node_id
                        )

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

    def store_nodes(self, nodes):
        # Add nodes to neo4j
        id_map = self.store_nodes_and_relationships(nodes)
        logging.info(f"Node ID Map: {id_map}")

        # Add nodes with embeddings to Qdrant
        self.add_nodes_to_qdrant(nodes, id_map)

    def close(self):
        self.neo4j_client.close()
