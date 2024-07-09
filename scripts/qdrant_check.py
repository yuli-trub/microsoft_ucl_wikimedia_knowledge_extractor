from qdrant_client import QdrantClient
from helper import load_env
from llama_index.core import VectorStoreIndex
from llama_index.core.storage import StorageContext

# Load environment variables
env_vars = load_env("QDRANT_PORT", "QDRANT_HOST")
QDRANT_PORT = env_vars["QDRANT_PORT"]
QDRANT_HOST = env_vars["QDRANT_HOST"]

# Connect to Qdrant
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# List all collections
collections = client.get_collections()
print("Collections in Qdrant:", collections)


# Check vectors in the collection
response = client.scroll(
    collection_name="spider_man_collection",
    limit=10,
)
print("Vectors in collection:", response)
