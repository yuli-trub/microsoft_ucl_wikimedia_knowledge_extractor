from dotenv import load_dotenv
import os
import pickle
import hashlib
import time
import logging


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
        "NEO4J_URI": os.getenv("NEO4J_URI"),
        "NEO4J_USER": os.getenv("NEO4J_USER"),
        "NEO4J_PASSWORD": os.getenv("NEO4J_PASSWORD"),
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


# log duration
def log_duration(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        logging.info(
            f"Function {func.__name__} took {duration:.2f} seconds to complete"
        )
        return result

    return wrapper
