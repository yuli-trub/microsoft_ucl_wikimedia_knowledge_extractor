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
from llama_index.core import Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.core import PromptTemplate

# import pipeline
from llama_ingestionator.pipeline import create_pipeline, run_pipeline

# import storage_manager
from storage.storage_manager import StorageManager
from storage.qdrant_setup import setup_qdrant_client

# import retriever
from retriever.retrievifier import GraphVectorRetriever


# TODO: fix the links with new API call

# logging config
logging.basicConfig(
    level=logging.INFO,
    encoding="utf-8",
    filename="app.log",
    format="%(asctime)s - %(levelname)s - %(message)s",
)

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
    "NEO4J_URI",
    "NEO4J_USER",
    "NEO4J_PASSWORD",
)

GPT4_ENDPOINT = f'{env_vars["OPENAI_ENDPOINT"]}/openai/deployments/{env_vars["GPT4O_DEPLOYMENT_ID"]}/chat/completions?api-version={env_vars["GPT4O_API_VERSION"]}'

# # Neo4j and Qdrant configurations
neo4j_config = {
    "uri": env_vars["NEO4J_URI"],
    "user": env_vars["NEO4J_USER"],
    "password": env_vars["NEO4J_PASSWORD"],
}

qdrant_config = {
    "host": env_vars["QDRANT_HOST"],
    "port": env_vars["QDRANT_PORT"],
    "collection_name": "squirrel",
}

# # Initialize StorageManager
storage_manager = StorageManager(neo4j_config, qdrant_config)

# set up embedding model
embed_model = AzureOpenAIEmbedding(
    model="text-embedding-ada-002",
    deployment_name=env_vars["EMBEDDING_DEPLOYMENT_ID"],
    api_key=env_vars["AZURE_OPENAI_API_KEY"],
    azure_endpoint=env_vars["OPENAI_ENDPOINT"],
    api_version=env_vars["EMBEDDING_API_VERSION"],
)

# set up llm
llm = AzureOpenAI(
    model="gpt-4o",
    deployment_name=env_vars["GPT4O_DEPLOYMENT_ID"],
    api_key=env_vars["AZURE_OPENAI_API_KEY"],
    azure_endpoint=GPT4_ENDPOINT,
    api_version=env_vars["GPT4O_API_VERSION"],
)
Settings.llm = llm

# get pre saved initial Nodes
filename = "squirrel_image_test.pkl"

if os.path.exists(filename):
    documents = load_documents_from_file(filename)
    print(f"Loaded {len(documents)} documents from {filename}")
else:
    documents = process_page_into_doc_and_nodes("Squirrel")
    save_documents_to_file(documents, filename)
    print(f"Processed and saved {len(documents)} documents")

# initialise the pipeline
pipeline = create_pipeline()

# get pre saved transformed Nodes
test_filename = "squirrel-pipeline-image_embed_test.pkl"

if os.path.exists(test_filename):
    test_nodes = load_documents_from_file(test_filename)
    print(f"Loaded {len(test_nodes)} documents from {test_filename}")
else:
    logging.info(f"Processing {len(documents)} documents")
    test_nodes = run_pipeline(documents, pipeline, embed_model)
    save_documents_to_file(test_nodes, test_filename)
    print(f"Processed and saved {len(test_nodes)} documents")

# Add nodes to neo4j
# id_map = storage_manager.store_nodes_and_relationships(test_nodes)
# logging.info(f"Node ID Map: {id_map}")

# # Add nodes with embeddings to Qdrant
# storage_manager.add_nodes_to_qdrant(test_nodes, id_map)


# Initialize vector store and index
(
    client,
    aclient,
    text_vector_store,
    image_vector_store,
    text_storage_context,
    image_storage_context,
) = setup_qdrant_client(**qdrant_config)

index = VectorStoreIndex.from_vector_store(text_vector_store, embed_model=embed_model)


# fusion retriever
from llama_index.core.retrievers import QueryFusionRetriever

# fusion_retriever = QueryFusionRetriever(
#     [text_index.as_retriever()],
#     similarity_top_k=10,
#     num_queries=4,
#     use_async=False,
#     verbose=True,
# )

# retrieval stage
retriever = GraphVectorRetriever(storage_manager, embed_model)


# Example question

question = "What is the main characteristics of squirrel?"

# search_results = fusion_retriever._run_sync_queries(question)
# logging.info(f"Search results: {len(search_results)}")


parent_nodes = retriever.fusion_retrieve(question)
logging.info(
    f"Retrieved parent nodes: { [ node.metadata['type'] for node in parent_nodes ] }"
)


from llama_index.core.schema import ImageNode

# # get text from parent nodes or images
images = [node.metadata["url"] for node in parent_nodes if isinstance(node, ImageNode)]
texts = [node.text for node in parent_nodes if hasattr(node, "text")]
combined_text = " ".join(texts)
combined_images = " ".join(images)
logging.info(f"Texts from parent nodes: {texts}")
logging.info(f"Images from parent nodes: {images}")


# # llm input
llm_rag_input = (
    "The following context information has been retrieved:\n\n"
    "Context:\n"
    f"{combined_text}\n"
    "\nImages:\n"
    f"{combined_images}\n"
    "\nBased on this information, please answer the following question in a detailed and informative manner:\n"
    f"Question: {question}\n"
    "Answer:"
)
llm_input = f"Question: {question}\n\nAnswer:"


# llm_rag_input = (
#     "Context information is below.\n"
#     "---------------------\n"
#     f"{combined_text}\n"
#     "---------------------\n"
#     "Given the context information and prior knowledge"
#     "answer the query.\n"
#     f"Query: {question}\n"
#     "Answer: "
# )

# llm_input = f"{question}, if you don't know just state it"


# qa_tmpl_str = (
#     "Context information is below.\n"
#     "---------------------\n"
#     "{context_str}\n"
#     "---------------------\n"
#     "Given the context information and prior knowledge, "
#     "answer the query.\n"
#     "Query: {query_str}\n"
#     "Answer: "
# )
# qa_tmpl = PromptTemplate(qa_tmpl_str)

query_engine = index.as_query_engine(multi_modal_llm=llm)

from llama_index.core.query_engine import FLAREInstructQueryEngine

flare_query_engine = FLAREInstructQueryEngine(
    query_engine=query_engine,
    max_iterations=7,
    verbose=True,
)

# # prompts_dict = query_engine.get_prompts()
# # logging.info(prompts_dict)

# # # Query with context

standard_response = llm.complete(llm_input)
enhanced_response = flare_query_engine.query(llm_rag_input)

# Log and print both responses for comparison
logging.info(f"Enhanced Response: {enhanced_response}")
logging.info(f"Standard Response: {standard_response}")

storage_manager.close()
