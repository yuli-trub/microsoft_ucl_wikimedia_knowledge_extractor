# config.py
import os
from scripts.helper import load_env
import logging


def get_env_vars():
    return load_env(
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


def setup_logging() -> None:
    """Set up logging configuration."""

    log_directory = os.path.expanduser('~/app_logs') 
    os.makedirs(log_directory, exist_ok=True) 

    log_file_path = os.path.join(log_directory, 'app.log')

    log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(log_formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)


def get_neo4j_config(env_vars):
    return {
        "uri": env_vars["DB_NEO4J_URI"],
        "user": env_vars["DB_NEO4J_USER"],
        "password": env_vars["DB_NEO4J_PASSWORD"],
    }


def get_qdrant_config(env_vars):
    return {
        "host": env_vars["QDRANT_HOST"],
        "port": env_vars["QDRANT_PORT"],
        "collection_name": env_vars["QDRANT_COLLECTION_NAME"],
    }
