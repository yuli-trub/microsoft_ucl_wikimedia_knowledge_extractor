from neo4j import GraphDatabase
from llama_index.core.schema import BaseNode, TextNode, ImageNode, Document
import json
import numpy as np
import logging
from scripts.helper import load_env
from llama_index.core.schema import RelatedNodeInfo


# TODO: adapt for image nodes as well

# logging = logging.getlogging()

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

    # check if db exists
    def database_exists(self, database_name: str) -> bool:
        with self.driver.session(database="system") as session:
            result = session.run(
                "SHOW DATABASES YIELD name WHERE name = $name RETURN name",
                name=database_name
            )
            return result.single() is not None
        
    # create db
    def create_database(self, database_name: str):
        with self.driver.session(database="system") as session:
            session.run(f"CREATE DATABASE {database_name}")
            logging.info(f"Database {database_name} created.")

    def start_database(self, database_name: str):
        with self.driver.session(database="system") as session:
            session.run(f"START DATABASE {database_name}")
            logging.info(f"Database {database_name} started.")

    # create a Document node
    def create_document_node(self, node: Document):
        node_data = node_to_metadata_dict(node)
        with self.driver.session() as session:
            neo_node_id = session.execute_write(self._create_document_node, node_data)
        logging.info(f"Document node created with neo4j ID: {neo_node_id}")
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
        RETURN elementId(d) AS neo_node_id
        """
        result = tx.run(query, **node_data)
        return result.single()["neo_node_id"]

    # create a text node
    def create_text_node(self, node: TextNode):
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
    def create_image_node(
        self,
        node: ImageNode,
    ):
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
        # logging.info(
        #     f"Creating query with these params {relationship_type} from {from_node_id} to {to_node_id}"
        # )
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
        # logging.info(f"Relationship created: {result}")
        return result.single()["r"]

    def get_node_by_neo_id(self, node_id: str) -> BaseNode:
        with self.driver.session() as session:
            record = session.execute_read(self._get_node, node_id)
            if record is None:
                logging.warning(f"Node with neo4j ID {node_id} not found.")
                return None
            node = metadata_dict_to_node(record)
            logging.info(f"Node retrieved with neo4j ID: {node_id}")
            return node

    @staticmethod
    def get_node_by_neo_id(tx, node_id):
        query = """
        MATCH (n) 
        WHERE elementId(n)=$node_id 
        RETURN n
        """
        result = tx.run(query, node_id=node_id).single()
        return result["n"] if result else None

    def get_node_by_llama_id(self, node_id: str):
        logging.info(f"Tryig to retrieve node with Llama ID: {node_id}")
        with self.driver.session() as session:
            record = session.execute_read(self._get_node, node_id)
            if record is None:
                logging.warning(f"Node with neo4j ID {node_id} not found.")
                return None
            node = metadata_dict_to_node(record)
            logging.info(f"Node retrieved with neo4j ID: {node_id}")
            return node

    @staticmethod
    def get_node_by_llama_id(tx, node_id):
        query = """
        MATCH (n) 
        WHERE n.llama_node_id=$node_id 
        RETURN n
        """
        result = tx.run(query, node_id=node_id).single()
        return result["n"] if result else None

    def get_parent_node(self, node_id: str):
        with self.driver.session() as session:
            logging.info(f"Retrieving parent node for Llama ID: {node_id}")
            record = session.execute_read(self._get_parent_node, node_id)
            if record is None:
                logging.warning(f"Parent node for Llama ID {node_id} not found.")
                return None

            parent_node = metadata_dict_to_node(record)
            logging.info(f"Parent node retrieved : {parent_node}")
            return parent_node

    @staticmethod
    def _get_parent_node(tx, node_id):
        query = """
        MATCH (parent)-[:PARENT]->(n {llama_node_id: $node_id})
        WHERE parent.type IN ['section', 'subsection', 'image', 'plot']
        RETURN parent
        """
        result = tx.run(query, node_id=node_id).single()
        return result["parent"] if result else None


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


# test it out
# neo4j_client = Neo4jClient(DB_NEO4J_URI, DB_NEO4J_USER, DB_NEO4J_PASSWORD)
