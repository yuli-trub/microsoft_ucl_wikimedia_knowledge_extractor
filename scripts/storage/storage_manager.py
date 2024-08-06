from llama_index.core.schema import Document, ImageNode, TextNode
import logging
from storage.graph_db_setup import Neo4jClient
from storage.qdrant_setup import setup_qdrant_client, add_node_to_qdrant
from qdrant_client.http.models import models, SearchRequest, SearchParams, VectorStruct
from llama_index.core import VectorStoreIndex


class StorageManager:
    def __init__(self, neo4j_config, qdrant_config):
        self.neo4j_client = Neo4jClient(**neo4j_config)
        self.qdrant_client, self.vector_store, self.storage_context = (
            setup_qdrant_client(**qdrant_config)
        )

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
                    logging.info(
                        f"Creating relationship {relationship_type} from {node.node_id} to {related_node_info.node_id}"
                    )
                    to_id = node.node_id
                    from_id = related_node_info.node_id
                    if to_id:
                        check_relationship = neo4j_client.create_relationship(
                            from_id, to_id, relationship_type
                        )
                        logging.info(f"Relationship created: {check_relationship}")

        logging.info("Nodes and relationships created in Neo4j")

        return node_id_map

    def add_nodes_to_qdrant(self, nodes, id_map):
        logging.info("Adding nodes with embeddings to Qdrant.")
        for node in nodes:
            if node.embedding is not None:
                neo_node_id = id_map.get(node.node_id)
                if neo_node_id:
                    add_node_to_qdrant(
                        self.vector_store, node, neo_node_id, node.node_id
                    )

    def build_index(self):
        logging.info("Building VectorStoreIndex from Qdrant vector store.")
        index = VectorStoreIndex.from_vector_store(self.vector_store)
        logging.info("Index built successfully.")
        return index

    def vector_search(self, query_vector, top_k):
        return self.qdrant_client.search(
            collection_name=self.vector_store.collection_name,
            query_vector=query_vector,
            with_payload=True,
            limit=top_k,
        )

    def close(self):
        self.neo4j_client.close()
