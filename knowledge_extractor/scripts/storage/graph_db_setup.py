from neo4j import GraphDatabase
from llama_index.core.schema import BaseNode, TextNode, ImageNode, Document
import json
import numpy as np
import logging
from scripts.helper import load_env


# get env vars
env_vars = load_env("DB_NEO4J_URI", "DB_NEO4J_USER", "DB_NEO4J_PASSWORD")
DB_NEO4J_URI = env_vars["DB_NEO4J_URI"]
DB_NEO4J_USER = env_vars["DB_NEO4J_USER"]
DB_NEO4J_PASSWORD = env_vars["DB_NEO4J_PASSWORD"]


class Neo4jClient:
    # initialise the db - connection
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    # close connection
    def close(self):
        self.driver.close()

    # create a Document node
    def create_document_node(self, node: Document):
        """ Create a document node in the graph db """
        node_data = node_to_metadata_dict(node)
        with self.driver.session() as session:
            neo_node_id = session.execute_write(self._create_document_node, node_data)
        logging.info(f"Document node created with neo4j ID: {neo_node_id}")
        return neo_node_id

    @staticmethod
    def _create_document_node(tx, node_data):
        """ Create a document node in the graph db with CYPHER query """
        query = """
        CREATE (d:Document {
            llama_node_id: $llama_node_id, 
            title: $title,
            type: $type,
            metadata: $metadata
        })
        RETURN elementId(d) AS neo_node_id
        """
        result = tx.run(query, **node_data)
        return result.single()["neo_node_id"]

    # create a text node
    def create_text_node(self, node: TextNode):
        """ Create a text node in the graph db """
        node_data = node_to_metadata_dict(node)
        label = node_data["type"].capitalize()
        with self.driver.session() as session:
            neo_node_id = session.execute_write(
                self._create_text_node, node_data, label
            )
        logging.info(f"Text Node created with neo4j ID: {neo_node_id}")
        return neo_node_id

    @staticmethod
    def _create_text_node(tx, node_data, label):
        """ Create a text node in the graph db with CYPHER query """
        query = f"""
        CREATE (n:{label} {{
            llama_node_id: $llama_node_id, 
            text: $text, 
            type: $type,
            title: $title,
            relationships: $relationships,
            metadata: $metadata, 
            embedding: $embedding
        }})
        RETURN elementId(n) AS neo_node_id, n.llama_node_id AS llama_node_id
        """
        result = tx.run(query, **node_data)
        return result.single()["neo_node_id"]

    # create an ImageNode
    def create_image_node(self, node: ImageNode):
        """ Create an image node in the graph db """
        node_data = node_to_metadata_dict(node)
        label = node_data["type"].capitalize()
        image_url = node.metadata["url"]
        with self.driver.session() as session:
            neo_node_id = session.execute_write(
                self._create_image_node, node_data, label, image_url
            )
        logging.info(f"ImageNode created with neo4j ID: {neo_node_id}")
        return neo_node_id

    @staticmethod
    def _create_image_node(tx, node_data, label, image_url):
        """ Create an image node in the graph db with CYPHER query """
        query = f"""
        CREATE (n:{label} {{
            llama_node_id: $llama_node_id, 
            image_url: "{image_url}", 
            metadata: $metadata, 
            title: $title,
            type: $type,
            relationships: $relationships, 
            embedding: $embedding
        }})
        RETURN elementId(n) AS neo_node_id, n.llama_node_id AS llama_node_id
        """
        result = tx.run(query, **node_data)
        return result.single()["neo_node_id"]

    def create_relationship(
        self, from_node_id: str, to_node_id: str, relationship_type: str
    ):
        """ Create a relationship between two nodes in the graph db

         Args:
            from_node_id (str): The ID of the node to create the relationship from
            to_node_id (str): The ID of the node to create the relationship to
            relationship_type (str): The type of relationship to create
        
        Returns:
            Relationship: The created relationship
        """
        try:
            with self.driver.session() as session:
                relationship = session.write_transaction(
                    self._create_relationship,
                    from_node_id,
                    to_node_id,
                    relationship_type,
                )
            logging.info(
                f"Created relationship {relationship_type} from {from_node_id} to {to_node_id}"
            )
            return relationship

        except Exception as e:
            logging.error(f"Failed to create relationship: {e}")

    @staticmethod
    def _create_relationship(tx, from_node_id, to_node_id, relationship_type):
        """ Create a relationship between two nodes in the graph db with CYPHER query """

        query = f"""
        MATCH (a {{llama_node_id: $from_node_id}}), (b {{llama_node_id: $to_node_id}})
        CREATE (a)-[r:{relationship_type}]->(b)
        RETURN r
        """

        result = tx.run(
            query,
            from_node_id=from_node_id,
            to_node_id=to_node_id,
            relationship_type=relationship_type,
        )
        return result.single()["r"]



# serialise node object
def node_to_metadata_dict(node: BaseNode) -> dict:
    """ Serialise a node object to a dictionary for storage in the graph db """
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
    ''' Deserialise a dictionary from the graph db to a node object '''
    # relationships = {
    #     key: RelatedNodeInfo(**value)
    #     for key, value in json.loads(meta.get("relationships", "{}")).items()
    # }

    metadata = json.loads(meta["metadata"])

    embedding = (
        np.array(meta["embedding"]).tolist()
        if meta.get("embedding") is not None
        else None
    )

    if "text" in meta:
        return TextNode(
            node_id=meta["llama_node_id"],
            text=meta["text"],
            metadata=json.loads(meta["metadata"]),
            # relationships=relationships,
            embedding=embedding,
        )
    if "image_url" in meta:
        return ImageNode(
            node_id=meta["llama_node_id"],
            image_url=meta["image_url"],
            metadata=json.loads(meta["metadata"]),
            # relationships=relationships,
            embedding=embedding,
        )
    elif "summary" in meta:
        metadata["summary"] = meta["summary"]
        return Document(
            node_id=meta["llama_node_id"],
            metadata=metadata,
            # relationships=relationships,
            embedding=embedding,
        )


