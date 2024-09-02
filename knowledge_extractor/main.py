from scripts.helper import (
    load_env,
    save_documents_to_file,
    load_documents_from_file,
    sanitise_filename
)
from scripts.llama_ingestionator.documentifier import process_page_into_doc_and_nodes
import logging
from scripts.initialiser import (
    initialise_embed_model,
    initialise_llm,
)
from scripts.config import (
    get_env_vars,
    setup_logging,
    get_neo4j_config,
    get_qdrant_config,
)

from scripts.wiki_crawler.searchinator import search_wiki
import os

# import pipeline
from scripts.llama_ingestionator.pipeline import create_pipeline, run_pipeline

# import storage_manager
from scripts.storage.storage_manager import StorageManager

# import data processing
from scripts.data_processing import get_initial_nodes, create_transformed_nodes



def main() -> None:
    setup_logging()

    try:

        # Load environment variables
        env_vars = get_env_vars()

        # Neo4j and Qdrant configurations
        neo4j_config = get_neo4j_config(env_vars)
        qdrant_config = get_qdrant_config(env_vars)

        # Set up embedding model
        embed_model = initialise_embed_model(env_vars)

        # Set up LLM
        llm = initialise_llm(env_vars)
        
        # Initialise StorageManager
        storage_manager = StorageManager(neo4j_config, qdrant_config)

        logging.info(env_vars["DOMAIN_TOPIC"])
        # === KNOWLEDGE EXTRACTOR PART ===

        topic = env_vars["DOMAIN_TOPIC"]
        num_pages = int(env_vars["NUM_WIKI_PAGES"])

        # Load or create initial nodes
        logging.info(f'topic: {topic}, num_pages: {num_pages}')
        initial_documents = get_initial_nodes(topic, num_pages)

        # Initialise the pipeline
        pipeline = create_pipeline()

        # Load or create transformed nodes
        # test_filename = "./data/squirrel-pipeline-image_embed_test.pkl"
        pipeline_transformed_nodes = create_transformed_nodes(
            initial_documents, env_vars['DOMAIN_TOPIC'], pipeline, embed_model
        )

        # Store nodes and relationships in Neo4j
        # uncoment later - already stored
        for page in pipeline_transformed_nodes:
            storage_manager.store_nodes(page)

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise
    finally:
        storage_manager.close()


if __name__ == "__main__":
    main()