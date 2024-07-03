# so combining..
# 1. create transformations
# 2. from documentifier call the function to create initial docs
# 3. create ingestion pipeline with
# document management  = docstore=SimpleDocumentStore()
# vector store??? -> add final output to vectore stor


from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from transformator import load_env
from documentifier import process_page_into_doc_and_nodes
from llama_index.core.ingestion import IngestionPipeline
from transformator import TextCleaner, SemanticChunkingTransformation
import re

# from llama_index.core.node_parser import SemanticSplitterNodeParser


# get env variables
(
    AZURE_OPENAI_API_KEY,
    OPENAI_ENDPOINT,
    GPT4O_DEPLOYMENT_ID,
    GPT4O_API_VERSION,
    GPT4_ENDPOINT,
    EMBEDDING_DEPLOYMENT_ID,
    EMBEDDING_API_VERSION,
) = load_env()


# set up embedding model
embed_model = AzureOpenAIEmbedding(
    model="text-embedding-ada-002",
    deployment_name=EMBEDDING_DEPLOYMENT_ID,
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=OPENAI_ENDPOINT,
    api_version=EMBEDDING_API_VERSION,
)

# vecto and doc store????

# get the docs and nodes
documents = process_page_into_doc_and_nodes("Python (programming language)")
print(len(documents))

# initialise the transformations
sem_chunk = SemanticChunkingTransformation()

pipeline = IngestionPipeline(
    transformations=[
        sem_chunk,
        TextCleaner(),
    ],
)


test_docs = documents[:5]
# nodes = sem_chunker(documents=test_docs, embed_model=embed_model)

nodes = pipeline.run(documents=test_docs)

if not nodes:
    print("The pipeline did not return any results.")
    exit()

# Print results
print("Transformed Documents:")
for i, doc in enumerate(nodes):
    print(f"Transformed Document {i+1}:")
    print(f"ID: {doc.metadata['id']}")
    print(f"Content: {doc.text[:100]}...\n")
