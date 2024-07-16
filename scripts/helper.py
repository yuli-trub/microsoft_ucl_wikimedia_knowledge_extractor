from dotenv import load_dotenv
import os
import pickle
import hashlib


# get env variables
def load_env(*keys):
    load_dotenv()
    env_vars = {
        "AZURE_OPENAI_API_KEY": os.getenv("AZURE_OPENAI_API_KEY"),
        "OPENAI_ENDPOINT": os.getenv("OPENAI_ENDPOINT"),
        "GPT4O_DEPLOYMENT_ID": os.getenv("GPT4O_DEPLOYMENT_ID"),
        "GPT4O_API_VERSION": os.getenv("GPT4O_API_VERSION"),
        "GPT4_ENDPOINT": f"{os.getenv('OPENAI_ENDPOINT')}/openai/deployments/{os.getenv('GPT4O_DEPLOYMENT_ID')}/chat/completions?api-version={os.getenv('GPT4O_API_VERSION')}",
        "EMBEDDING_DEPLOYMENT_ID": os.getenv("EMBEDDING_DEPLOYMENT_ID"),
        "EMBEDDING_API_VERSION": os.getenv("EMBEDDING_API_VERSION"),
        "QDRANT_PORT": os.getenv("QDRANT_PORT"),
        "QDRANT_HOST": os.getenv("QDRANT_HOST"),
    }
    return {key: env_vars[key] for key in keys}


# load documents
def save_documents_to_file(documents, filename="documents.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(documents, f)


def load_documents_from_file(filename="documents.pkl"):
    with open(filename, "rb") as f:
        return pickle.load(f)


# generate hash
def generate_document_hash(document):
    document_str = str(document.metadata) + document.text
    return hashlib.sha256(document_str.encode("utf-8")).hexdigest()
