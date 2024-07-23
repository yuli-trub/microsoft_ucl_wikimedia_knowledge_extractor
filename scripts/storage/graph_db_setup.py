from neo4j import GraphDatabase
from llama_index.core.schema import BaseNode, TextNode, ImageNode, Document
import json
import numpy as np
import logging
from helper import load_env
from llama_index.core.schema import RelatedNodeInfo


# TODO: adapt for image nodes as well

# setup logging
graph_db_logger = logging.getLogger("graph_db")


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

    # create a Document node
    def create_document_node(self, node: Document):
        node_data = node_to_metadata_dict(node)
        graph_db_logger.info(
            f"Creating Document Node with data: {node_data['metadata']}"
        )

        with self.driver.session() as session:
            neo_node_id = session.execute_write(self._create_document_node, node_data)
        graph_db_logger.info(f"Document node created with neo4j ID: {neo_node_id}")
        return neo_node_id

    @staticmethod
    def _create_document_node(tx, node_data):
        query = """
        CREATE (d:Document {
            llama_node_id: $llama_node_id, 
            title: $title,
            type: $type,
            metadata: $metadata
        })
        RETURN id(d) AS neo_node_id
        """
        result = tx.run(query, **node_data)
        return result.single()["neo_node_id"]

    # create a text node
    def create_text_node(self, node: TextNode):
        node_data = node_to_metadata_dict(node)
        # graph_db_logger.info(f"Creating Text Node with data: {node_data}")
        with self.driver.session() as session:
            neo_node_id = session.execute_write(self._create_text_node, node_data)
        graph_db_logger.info(f"Text Node created with neo4j ID: {neo_node_id}")
        return neo_node_id

    @staticmethod
    def _create_text_node(tx, node_data):
        query = """
        CREATE (n:TextNode {
            llama_node_id: $llama_node_id, 
            text: $text, 
            type: $type,
            title: $title,
            
            metadata: $metadata, 
            embedding: $embedding
        })
        RETURN id(n) AS neo_node_id
        """
        result = tx.run(query, **node_data)
        return result.single()["neo_node_id"]

    # create an ImageNode
    def create_image_node(self, node: ImageNode):
        node_data = node_to_metadata_dict(node)
        with self.driver.session() as session:
            neo_node_id = session.execute_write(self._create_image_node, node_data)
        graph_db_logger.info(f"ImageNode created with neo4j ID: {neo_node_id}")
        return neo_node_id

    @staticmethod
    def _create_image_node(tx, node_data):
        query = """
        CREATE (n:ImageNode {
            node_id: $node_id, 
            image_path: $image_path, 
            metadata: $metadata, 
            title: $title,
            type: $type,
            relationships: $relationships, 
            embedding: $embedding
        })
        RETURN id(n) AS neo_node_id
        """
        result = tx.run(query, **node_data)
        return result.single()["neo_node_id"]

    def create_relationship(
        self, from_node_id: str, to_node_id: str, relationship_type: str
    ):
        try:
            with self.driver.session() as session:
                session.write_transaction(
                    self._create_relationship,
                    from_node_id,
                    to_node_id,
                    relationship_type,
                )
            graph_db_logger.info(
                f"Created relationship {relationship_type} from {from_node_id} to {to_node_id}"
            )
        except Exception as e:
            graph_db_logger.error(f"Failed to create relationship: {e}")

    @staticmethod
    def _create_relationship(tx, from_node_id, to_node_id, relationship_type):
        query = """
        MATCH (a), (b)
        WHERE ID(a)={from_node_id} AND ID(b)={to_node_id}
        CREATE (a)-[r:{relationship_type}]->(b)
        RETURN r
        """
        tx.run(
            query,
            from_node_id=from_node_id,
            to_node_id=to_node_id,
            relationship_type=relationship_type,
        )

    def get_node(self, node_id: str) -> BaseNode:
        with self.driver.session() as session:
            record = session.execute_read(self._get_node, node_id)
            if record is None:
                graph_db_logger.warning(f"Node with neo4j ID {node_id} not found.")
                return None
            node = metadata_dict_to_node(record)
            graph_db_logger.info(f"Node retrieved with neo4j ID: {node_id}")
            return node

    @staticmethod
    def _get_node(tx, node_id):
        query = "MATCH (n {node_id: $node_id}) RETURN n"
        result = tx.run(query, node_id=node_id).single()
        return result["n"] if result else None


# serialise node object
def node_to_metadata_dict(node: BaseNode) -> dict:
    relationships = (
        {key: {"node_id": value.node_id} for key, value in node.relationships.items()}
        if node.relationships
        else {}
    )

    embedding = (
        node.embedding
        if isinstance(node.embedding, list)
        else node.embedding.tolist() if node.embedding is not None else None
    )

    base_data = {
        "llama_node_id": node.node_id,
        "metadata": json.dumps(node.metadata),
        "relationships": json.dumps(relationships),
        "embedding": embedding,
        "title": node.metadata.get("title", ""),
        "source": node.metadata.get("source", ""),
        "type": node.metadata.get("type", ""),
    }

    if isinstance(node, TextNode):
        base_data["text"] = node.text
    elif isinstance(node, ImageNode):
        base_data["image_path"] = node.image_path
    elif isinstance(node, Document):
        base_data["summary"] = node.metadata.get("summary", "")

    return base_data


# deserialise dict from graph db
def metadata_dict_to_node(meta: dict) -> BaseNode:
    relationships = {
        key: RelatedNodeInfo(**value)
        for key, value in json.loads(meta.get("relationships", "{}")).items()
    }

    metadata = json.loads(meta["metadata"])

    if "text" in meta:
        return TextNode(
            node_id=meta["llama_node_id"],
            text=meta["text"],
            metadata=json.loads(meta["metadata"]),
            relationships=relationships,
            embedding=(
                np.array(meta["embedding"]) if meta["embedding"] is not None else None
            ),
        )
    if "image_path" in meta:
        return ImageNode(
            node_id=meta["llama_node_id"],
            image_path=meta["image_path"],
            metadata=json.loads(meta["metadata"]),
            relationships=relationships,
            embedding=(
                np.array(meta["embedding"]) if meta["embedding"] is not None else None
            ),
        )
    elif "summary" in meta:
        metadata["summary"] = meta["summary"]
        return Document(
            node_id=meta["llama_node_id"],
            metadata=metadata,
            relationships=relationships,
            embedding=(
                np.array(meta["embedding"]) if meta["embedding"] is not None else None
            ),
        )
    else:
        return BaseNode(node_id=meta["node_id"], metadata=json.loads(meta["metadata"]))


# test it out
# neo4j_client = Neo4jClient(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)