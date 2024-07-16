from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from helper import (
    load_env,
    log_duration,
    save_documents_to_file,
    load_documents_from_file,
)
from documentifier import process_page_into_doc_and_nodes
from llama_index.core.ingestion import IngestionPipeline, IngestionCache
from transformator import (
    TextCleaner,
    SemanticChunkingTransformation,
    EntityExtractorTransformation,
    SummaryTransformation,
    KeyTakeawaysTransformation,
    EmbeddingTransformation,
)
import os
import logging
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.core import Settings, VectorStoreIndex

# import qdrant_client
from qdrant_setup import setup_qdrant_client

# TODO: do caching with remote chache management????
from llama_index.storage.kvstore.redis import RedisKVStore as RedisCache


# config logging
logging.basicConfig(
    level=logging.INFO,
    filename="ingestionator.log",
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


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


# Set up Qdrant client
client, vector_store, storage_context = setup_qdrant_client(
    host=QDRANT_HOST, port=QDRANT_PORT, collection_name="napoleon_collection"
)

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


filename = "napoleon-documents.pkl"

if os.path.exists(filename):
    documents = load_documents_from_file(filename)
    print(f"Loaded {len(documents)} documents from {filename}")
else:
    documents = process_page_into_doc_and_nodes("Spider-Man")
    save_documents_to_file(documents, filename)
    print(f"Processed and saved {len(documents)} documents")

# cache - TODO - figure out remote cache with Redis
# ingest_cache = IngestionCache(
#     cache=RedisCache.from_host_and_port(host="127.0.0.1", port=6379),
#     collection="my_test_cache",
# )

# initialise the transformations
text_cleaner = TextCleaner()
semantic_chunking = SemanticChunkingTransformation()
entities_extractor = EntityExtractorTransformation()
summarisor = SummaryTransformation()
key_takeaways = KeyTakeawaysTransformation()
embedding = EmbeddingTransformation()

# ingestion pipeline
pipeline = IngestionPipeline(
    transformations=[
        semantic_chunking,
        text_cleaner,
        entities_extractor,
        summarisor,
        key_takeaways,
        embedding,
    ],
    # cache=ingest_cache,
    vector_store=vector_store,
)


test_docs = documents[:2]


# run the pipeline
@log_duration
def run_pipeline(documents, embed_model):
    return pipeline.run(documents=documents, text_embed_model=embed_model)


nodes = run_pipeline(test_docs, embed_model)


# Run the pipeline and check cache
# for doc in test_docs:
#     doc_hash = generate_document_hash(doc)

#     cached_result = ingest_cache.get(doc_hash)
#     print(cached_result)

#     if cached_result:
#         logging.info(f"Using cached result for document ID: {doc.metadata['id']}")
#         nodes = cached_result
#     else:
#         logging.info(f"Processing new document ID: {doc.metadata['id']}")
#         nodes = pipeline.run(documents=[doc])
#         logging.info(f"Storing result in cache for document ID: {doc.metadata['id']}")
#         ingest_cache.put(doc_hash, nodes)
#         # check the cache?
#         cached_result_after_put = ingest_cache.get(doc_hash)
#         logging.info(f"Cached result after put: {cached_result_after_put}")

# if not nodes:
#     logging.error("The pipeline did not return any results.")
#     exit()

# Print results
# print("Transformed Documents:")
# for i, doc in enumerate(nodes):
#     print(f"Transformed Document {i+1}:")
#     print(f"ID: {doc.metadata['id']}")
#     print(f"Content: {doc.text[:200]}...\n")


# Load the index from the storage context
index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store, embed_model=embed_model
)


# # Test query
@log_duration
def test_query(index):
    query_engine = index.as_query_engine(similarity_top_k=8)
    response = query_engine.query("What color was the Napoleon's white horse?")
    logger.info(f"Query Response: {response}")
    logger.info(f"Query Response Length: {len(response.source_nodes)}")
    return response


response = test_query(index)
print("Query Response:", response)
