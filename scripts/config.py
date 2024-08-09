# config.py
import os
from helper import load_env
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
        "NEO4J_URI",
        "NEO4J_USER",
        "NEO4J_PASSWORD",
    )


def setup_logging() -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        encoding="utf-8",
        filename="app.log",
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def get_neo4j_config(env_vars):
    return {
        "uri": env_vars["NEO4J_URI"],
        "user": env_vars["NEO4J_USER"],
        "password": env_vars["NEO4J_PASSWORD"],
    }


def get_qdrant_config(env_vars):
    return {
        "host": env_vars["QDRANT_HOST"],
        "port": env_vars["QDRANT_PORT"],
        "collection_name": env_vars["QDRANT_COLLECTION_NAME"],
    }
