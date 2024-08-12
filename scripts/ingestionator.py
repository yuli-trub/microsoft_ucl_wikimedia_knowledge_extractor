from scripts.helper import (
    load_env,
    save_documents_to_file,
    load_documents_from_file,
)
from llama_ingestionator.documentifier import process_page_into_doc_and_nodes
import os
import logging
from initialiser import (
    initialise_embed_model,
    initialise_llm,
)
from config import setup_logging, get_neo4j_config, get_qdrant_config

# import pipeline
from llama_ingestionator.pipeline import create_pipeline, run_pipeline

# import storage_manager
from scripts.storage.storage_manager import StorageManager

# import retriever
from retriever.retrievifier import GraphVectorRetriever

from llama_index.core.query_engine import FLAREInstructQueryEngine


# TODO: fix the links with new API call

# logging config
setup_logging()

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
    "QDRANT_COLLECTION_NAME",
    "DB_NEO4J_URI",
    "DB_NEO4J_USER",
    "DB_NEO4J_PASSWORD",
)

GPT4_ENDPOINT = f'{env_vars["OPENAI_ENDPOINT"]}/openai/deployments/{env_vars["GPT4O_DEPLOYMENT_ID"]}/chat/completions?api-version={env_vars["GPT4O_API_VERSION"]}'

# # Neo4j and Qdrant configurations
neo4j_config = get_neo4j_config(env_vars)
qdrant_config = get_qdrant_config(env_vars)

# # Initialise StorageManager
storage_manager = StorageManager(neo4j_config, qdrant_config)

# set up embedding model
embed_model = initialise_embed_model(env_vars)

# set up llm
llm = initialise_llm(env_vars)


def get_initial_nodes(filename):
    if os.path.exists(filename):
        documents = load_documents_from_file(filename)
        print(f"Loaded {len(documents)} documents from {filename}")
    else:
        documents = process_page_into_doc_and_nodes("Squirrel")
        save_documents_to_file(documents, filename)
        print(f"Processed and saved {len(documents)} documents")
    return documents


def create_transformed_nodes(documents, file_name):
    if os.path.exists(file_name):
        pipeline_transformed_nodes = load_documents_from_file(file_name)
        print(f"Loaded {len(pipeline_transformed_nodes)} documents from {file_name}")
    else:
        logging.info(f"Processing {len(documents)} documents")
        pipeline_transformed_nodes = run_pipeline(documents, pipeline, embed_model)
        save_documents_to_file(pipeline_transformed_nodes, file_name)
        print(f"Processed and saved {len(pipeline_transformed_nodes)} documents")
    return pipeline_transformed_nodes


# get pre saved initial Nodes
filename = "squirrel_image_test.pkl"
initial_documents = get_initial_nodes(filename)

# initialise the pipeline
pipeline = create_pipeline()

# get pre saved transformed Nodes
test_filename = "squirrel-pipeline-image_embed_test.pkl"
pipeline_transformed_nodes = create_transformed_nodes(initial_documents, test_filename)

# store nodes and relationships in Neo4j
# storage_manager.store_nodes(pipeline_transformed_nodes)

# build index
index = storage_manager.build_index()

# retrieval stage
retriever = GraphVectorRetriever(storage_manager, embed_model)

# Example question
question = "What is the main characteristics of squirrel?"

parent_nodes = retriever.fusion_retrieve(question)
logging.info(
    f"Retrieved parent nodes: { [ node.metadata['type'] for node in parent_nodes ] }"
)

# get context from retrieved nodes
combined_text, combined_images = retriever.get_context_from_retrived_nodes(parent_nodes)

# # llm input
llm_rag_input = (
    "You are provided with context information retrieved from various sources. "
    "Please use this information to answer the following question thoroughly and concisely.\n\n"
    "Context (Textual Information):\n"
    f"{combined_text}\n"
    "\nRelevant Images (if applicable):\n"
    f"{combined_images}\n"
    "\nInstructions:\n"
    "- Base your answer primarily on the textual context provided.\n"
    "- Use relevant details from the images only if they add value to the answer.\n"
    "- Structure your response using headings and bullet points for clarity.\n"
    "- Avoid repeating information.\n"
    "- Ensure the answer is informative and directly addresses the question.\n\n"
    f"Question: {question}\n"
    "Answer:"
)
llm_input = f"Question: {question}\n\nAnswer:"

query_engine = index.as_query_engine(multi_modal_llm=llm)

flare_query_engine = FLAREInstructQueryEngine(
    query_engine=query_engine,
    max_iterations=4,
    verbose=True,
)

# # # Query with context

standard_response = llm.complete(llm_input)
enhanced_response = flare_query_engine.query(llm_rag_input)

# Log and print both responses for comparison
logging.info(f"Enhanced Response: {enhanced_response}")
logging.info(f"Standard Response: {standard_response}")

storage_manager.close()
