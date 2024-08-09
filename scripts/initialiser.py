# initialiser.py
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.core import Settings
from storage.storage_manager import StorageManager


def initialise_embed_model(env_vars):
    embed_model = AzureOpenAIEmbedding(
        model="text-embedding-ada-002",
        deployment_name=env_vars["EMBEDDING_DEPLOYMENT_ID"],
        api_key=env_vars["AZURE_OPENAI_API_KEY"],
        azure_endpoint=env_vars["OPENAI_ENDPOINT"],
        api_version=env_vars["EMBEDDING_API_VERSION"],
    )
    Settings.embed_model = embed_model
    return embed_model


def initialise_llm(env_vars):
    GPT4_ENDPOINT = f'{env_vars["OPENAI_ENDPOINT"]}/openai/deployments/{env_vars["GPT4O_DEPLOYMENT_ID"]}/chat/completions?api-version={env_vars["GPT4O_API_VERSION"]}'
    llm = AzureOpenAI(
        model="gpt-4o",
        deployment_name=env_vars["GPT4O_DEPLOYMENT_ID"],
        api_key=env_vars["AZURE_OPENAI_API_KEY"],
        azure_endpoint=GPT4_ENDPOINT,
        api_version=env_vars["GPT4O_API_VERSION"],
    )
    Settings.llm = llm
    return llm


def initialise_storage_manager(neo4j_config, qdrant_config):
    return StorageManager(neo4j_config, qdrant_config)
