from llama_index.core.schema import Document, ImageNode, TextNode
import logging


def store_nodes_and_relationships(test_nodes, neo4j_client):
    logging.info(f"nodes: {(len(test_nodes))}")
    node_id_map = {}

    for node in test_nodes:
        if isinstance(node, Document):
            logging.info(f"Creating document node {node.doc_id}")
            neo_node_id = neo4j_client.create_document_node(node)
            node_id_map[node.doc_id] = neo_node_id
        elif isinstance(node, TextNode) and not isinstance(node, Document):
            logging.info(f"Creating text node {node.node_id}")
            neo_node_id = neo4j_client.create_text_node(node)
            node_id_map[node.node_id] = neo_node_id
        elif isinstance(node, ImageNode):
            # logging.info(f"Creating image node {node.node_id}")
            neo_node_id = neo4j_client.create_image_node(node)
            node_id_map[node.node_id] = neo_node_id

    logging.info(f"Nodes created in Neo4j {node_id_map}")

    for node in test_nodes:
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
