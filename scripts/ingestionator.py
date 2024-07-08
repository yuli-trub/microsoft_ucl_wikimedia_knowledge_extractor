from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from transformator import load_env
from documentifier import process_page_into_doc_and_nodes
from llama_index.core.ingestion import IngestionPipeline
from transformator import (
    TextCleaner,
    SemanticChunkingTransformation,
    EntityExtractorTransformation,
    SummaryTransformation,
)
import re
import os

# just for now so i don't have to wait for the wiki data
import pickle


def save_documents_to_file(documents, filename="documents.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(documents, f)


def load_documents_from_file(filename="documents.pkl"):
    with open(filename, "rb") as f:
        return pickle.load(f)


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
# documents = process_page_into_doc_and_nodes("Spider-Man")
# print(len(documents))


# temporary local storage to reduce wiki fetching
filename = "documents.pkl"
if os.path.exists(filename):
    documents = load_documents_from_file(filename)
    print(f"Loaded {len(documents)} documents from {filename}")
else:
    documents = process_page_into_doc_and_nodes("Spider-Man")
    save_documents_to_file(documents, filename)
    print(f"Processed and saved {len(documents)} documents")


# initialise the transformations
semantic_chunking = SemanticChunkingTransformation()
entities_extractor = EntityExtractorTransformation()
summarisor = SummaryTransformation()

pipeline = IngestionPipeline(
    transformations=[semantic_chunking, TextCleaner(), entities_extractor, summarisor],
)


test_docs = documents[:2]


# run the pipeline
nodes = pipeline.run(documents=test_docs)

# test the transformation by itself
# nodes = summarisor(test_docs)

if not nodes:
    print("The pipeline did not return any results.")
    exit()

# Print results
print("Transformed Documents:")
for i, doc in enumerate(nodes):
    print(f"Transformed Document {i+1}:")
    print(f"ID: {doc.metadata['id']}")
    print(f"Content: {doc.text}...\n")
