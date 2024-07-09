from dotenv import load_dotenv
import os


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
