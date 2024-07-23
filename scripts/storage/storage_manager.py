from llama_index.core.schema import Document, ImageNode, TextNode
from logging_config import setup_logging
import logging


setup_logging()
logger = logging.getLogger("storage_manager")


def store_nodes_and_relationships(test_nodes, neo4j_client):
    node_id_map = {}

    for node in test_nodes:
        if isinstance(node, Document):
            logger.info(f"Creating document node {node.doc_id}")
            neo_node_id = neo4j_client.create_document_node(node)
            node_id_map[node.doc_id] = neo_node_id
        elif isinstance(node, TextNode) and not isinstance(node, Document):
            logger.info(f"Creating text node {node.node_id}")
            neo_node_id = neo4j_client.create_text_node(node)
            node_id_map[node.node_id] = neo_node_id
        elif isinstance(node, ImageNode):
            logger.info(f"Creating image node {node.node_id}")
            neo_node_id = neo4j_client.create_image_node(node)
            node_id_map[node.node_id] = neo_node_id

    logger.info(f"Nodes created in Neo4j {node_id_map}")

    for node in test_nodes:
        if node.relationships:
            logger.info(f"Creating relationships for node {node.node_id}")
            for relationship, related_node_info in node.relationships.items():
                from_id = node.node_id
                to_id = node_id_map.get(related_node_info.node_id)
                if to_id:
                    neo4j_client.create_relationship(from_id, to_id, relationship)

    logger.info("Nodes and relationships created in Neo4j")
