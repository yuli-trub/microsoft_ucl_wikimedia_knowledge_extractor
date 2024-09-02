from dotenv import load_dotenv
import os
import pickle
import hashlib
import time
import logging
import re


# get env variables
def load_env(*keys):
    """ Load environment variables """
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
        "QDRANT_COLLECTION_NAME": os.getenv("QDRANT_COLLECTION_NAME"),
        "DB_NEO4J_URI": os.getenv("DB_NEO4J_URI"),
        "DB_NEO4J_USER": os.getenv("DB_NEO4J_USER"),
        "DB_NEO4J_PASSWORD": os.getenv("DB_NEO4J_PASSWORD"),
        "COMPUTER_VISION_ENDPOINT": os.getenv("COMPUTER_VISION_ENDPOINT"),
        "COMPUTER_VISION_API_KEY": os.getenv("COMPUTER_VISION_API_KEY"),
        "DOMAIN_TOPIC": os.getenv("DOMAIN_TOPIC"),
        "NUM_WIKI_PAGES": os.getenv("NUM_WIKI_PAGES"),
    }
    return {key: env_vars[key] for key in keys}


# load documents
def save_documents_to_file(documents, filename):
    ""' Save documents to a file '""
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, "wb") as f:
        pickle.dump(documents, f)


def load_documents_from_file(filename):
    """ Load documents from a file """
    with open(filename, "rb") as f:
        return pickle.load(f)



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


def sanitise_filename(filename):
    """
    Sanitise a string to be used as a filename

    Parameters
        filename : str
            The string to sanitise

    Returns
            safe_filename : str
                The sanitised filename
    """
    invalid_chars = r'<>:,"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, "-")

    # replace -- with single -
    filename = re.sub(r"[\s-]+", "-", filename)

    return filename