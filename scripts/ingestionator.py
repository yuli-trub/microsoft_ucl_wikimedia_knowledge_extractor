from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from helper import (
    load_env,
    log_duration,
    save_documents_to_file,
    load_documents_from_file,
)
from llama_ingestionator.documentifier import process_page_into_doc_and_nodes
import os
import logging
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.core import Settings, VectorStoreIndex

from llama_index.core.schema import TextNode, ImageNode, Document

# import pipeline
from llama_ingestionator.pipeline import create_pipeline, run_pipeline

# import qdrant_client
from storage.qdrant_setup import setup_qdrant_client

# import neo4j_client
from storage.graph_db_setup import Neo4jClient


# Import the logging configuration
from logging_config import setup_logging

# Setup logging
logger = logging.getLogger("ingestionator")


# get env variables
env_vars = load_env(
    "AZURE_OPENAI_API_KEY",
    "OPENAI_ENDPOINT",
    "GPT4O_DEPLOYMENT_ID",
    "GPT4O_API_VERSION",
    "EMBEDDING_DEPLOYMENT_ID",
    "EMBEDDING_API_VERSION",
    "QDRANT_PORT",
    "QDRANT_HOST",
    "NEO4J_URI",
    "NEO4J_USER",
    "NEO4J_PASSWORD",
)

AZURE_OPENAI_API_KEY = env_vars["AZURE_OPENAI_API_KEY"]
OPENAI_ENDPOINT = env_vars["OPENAI_ENDPOINT"]
GPT4O_DEPLOYMENT_ID = env_vars["GPT4O_DEPLOYMENT_ID"]
GPT4O_API_VERSION = env_vars["GPT4O_API_VERSION"]
EMBEDDING_DEPLOYMENT_ID = env_vars["EMBEDDING_DEPLOYMENT_ID"]
EMBEDDING_API_VERSION = env_vars["EMBEDDING_API_VERSION"]
QDRANT_PORT = env_vars["QDRANT_PORT"]
QDRANT_HOST = env_vars["QDRANT_HOST"]
GPT4_ENDPOINT = f"{OPENAI_ENDPOINT}/openai/deployments/{GPT4O_DEPLOYMENT_ID}/chat/completions?api-version={GPT4O_API_VERSION}"
NEO4J_URI = env_vars["NEO4J_URI"]
NEO4J_USER = env_vars["NEO4J_USER"]
NEO4J_PASSWORD = env_vars["NEO4J_PASSWORD"]


# Set up Qdrant client
client, vector_store, storage_context = setup_qdrant_client(
    host=QDRANT_HOST, port=QDRANT_PORT, collection_name="napoleon_collection"
)

# initialise neo4j client
neo4j_client = Neo4jClient(uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD)

# set up embedding model
embed_model = AzureOpenAIEmbedding(
    model="text-embedding-ada-002",
    deployment_name=EMBEDDING_DEPLOYMENT_ID,
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=OPENAI_ENDPOINT,
    api_version=EMBEDDING_API_VERSION,
)

# set up llm
llm = AzureOpenAI(
    model="gpt-4o",
    deployment_name=GPT4O_DEPLOYMENT_ID,
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=GPT4_ENDPOINT,
    api_version=GPT4O_API_VERSION,
)
Settings.llm = llm

# get pre saved initial Nodes
filename = "llama_ingestionator/napoleon-documents.pkl"

if os.path.exists(filename):
    documents = load_documents_from_file(filename)
    print(f"Loaded {len(documents)} documents from {filename}")
else:
    documents = process_page_into_doc_and_nodes("Napoleon")
    save_documents_to_file(documents, filename)
    print(f"Processed and saved {len(documents)} documents")


# run pipeline
test_docs = documents[:3]
pipeline = create_pipeline(vector_store)
test_nodes = run_pipeline(test_docs, pipeline, embed_model)


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

    return node_id_map


id_map = store_nodes_and_relationships(test_nodes, neo4j_client)
logger.info(f"Node ID Map: {id_map}")

# log results
# for doc in enumerate(test_nodes):
#     logger.info(f"ID: {doc[1].metadata['title']}")
#     logger.info(f"metadata: {doc[1].metadata}")
#     if doc[1].embedding is not None:
#         logger.info(f"embedding: {doc[1].embedding[:100]}")
#     logger.info(f"Content: {doc[1].text[:200]}...\n")


# Load the index from the storage context
# index = VectorStoreIndex.from_vector_store(
#     vector_store=vector_store, embed_model=embed_model
# )


# # # Test query
# @log_duration
# def test_query(index):
#     query_engine = index.as_query_engine(similarity_top_k=8)
#     response = query_engine.query("Where was Napoleon born?")
#     logger.info(f"Query Response: {response}")
#     logger.info(f"Query Response nodes: {(response.source_nodes)}")
#     return response


# response = test_query(index)
# print("Query Response:", response)
