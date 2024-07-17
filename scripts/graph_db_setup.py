from neo4j import GraphDatabase
from llama_index.core.schema import BaseNode, TextNode
import json
import numpy as np
import logging
from helper import load_env


# TODO: adapt for image nodes as well

# setup logging
graph_db_logger = logging.getLogger("graph_db")
graph_db_handler = logging.FileHandler("graph_db.log")
graph_db_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
graph_db_handler.setFormatter(graph_db_formatter)
graph_db_logger.addHandler(graph_db_handler)
graph_db_logger.setLevel(logging.INFO)

# get env vars
env_vars = load_env("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD")
NEO4J_URI = env_vars["NEO4J_URI"]
NEO4J_USER = env_vars["NEO4J_USER"]
NEO4J_PASSWORD = env_vars["NEO4J_PASSWORD"]


class Neo4jClient:
    # initialise the db - connection
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    # close connection
    def close(self):
        self.driver.close()

    # create a node in neo4j
    def create_node(self, node: BaseNode):
        node_data = node_to_metadata_dict(node)
        with self.driver.session() as session:
            session.execute_write(self._create_node, node_data)
        graph_db_logger.info(f"Node created with ID: {node.node_id}")

    @staticmethod
    def _create_node(tx, node_data):
        query = """
        CREATE (n:TextNode {
            node_id: $node_id, 
            text: $text, 
            metadata: $metadata, 
            relationships: $relationships, 
            embedding: $embedding
        })
        """
        tx.run(query, **node_data)

    def get_node(self, node_id: str) -> BaseNode:
        with self.driver.session() as session:
            record = session.execute_read(self._get_node, node_id)
            if record is None:
                graph_db_logger.warning(f"Node with ID {node_id} not found.")
                return None
            node = metadata_dict_to_node(record)
            graph_db_logger.info(f"Node retrieved with ID: {node_id}")
            return node

    @staticmethod
    def _get_node(tx, node_id):
        query = "MATCH (n:TextNode {node_id: $node_id}) RETURN n"
        result = tx.run(query, node_id=node_id).single()
        return result["n"] if result else None


# serialise node object
def node_to_metadata_dict(node: BaseNode) -> dict:
    return {
        "node_id": node.node_id,
        "text": node.text,
        "metadata": json.dumps(node.metadata),
        "relationships": json.dumps(node.relationships),
        "embedding": node.embedding.tolist() if node.embedding is not None else None,
    }


# deserialise dict from graph db
def metadata_dict_to_node(meta: dict) -> BaseNode:
    return BaseNode(
        node_id=meta["node_id"],
        text=meta["text"],
        metadata=json.loads(meta["metadata"]),
        relationships=json.loads(meta["relationships"]),
        embedding=(
            np.array(meta["embedding"]) if meta["embedding"] is not None else None
        ),
    )


# test it out
neo4j_client = Neo4jClient(
    "bolt://localhost:7687", "neo4j", "absurd-gram-meaning-senior-bagel-5689"
)
