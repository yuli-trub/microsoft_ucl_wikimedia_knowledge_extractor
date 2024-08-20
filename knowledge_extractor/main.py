from scripts.helper import (
    load_env,
    save_documents_to_file,
    load_documents_from_file,
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
import os

# import pipeline
from scripts.llama_ingestionator.pipeline import create_pipeline, run_pipeline

# import storage_manager
from scripts.storage.storage_manager import StorageManager



def get_initial_nodes(filename: str) -> list:
    """Load initial nodes from a file or process a page to create them."""
    try:
        if os.path.exists(filename):
            documents = load_documents_from_file(filename)
            logging.info(f"Loaded {len(documents)} documents from {filename}")
        else:
            documents = process_page_into_doc_and_nodes("Squirrel")
            save_documents_to_file(documents, filename)
            logging.info(f"Processed and saved {len(documents)} documents")
    except Exception as e:
        logging.error(f"Failed to get initial nodes: {e}")
        raise
    return documents


def create_transformed_nodes(
    documents: list, file_name: str, pipeline, embed_model
) -> list:
    """Transform and save nodes using the pipeline."""
    try:
        if os.path.exists(file_name):
            pipeline_transformed_nodes = load_documents_from_file(file_name)
            logging.info(
                f"Loaded {len(pipeline_transformed_nodes)} documents from {file_name}"
            )
        else:
            logging.info(f"Processing {len(documents)} documents")
            pipeline_transformed_nodes = run_pipeline(documents, pipeline, embed_model)
            save_documents_to_file(pipeline_transformed_nodes, file_name)
            logging.info(
                f"Processed and saved {len(pipeline_transformed_nodes)} documents"
            )
    except Exception as e:
        logging.error(f"Failed to create transformed nodes: {e}")
        raise
    return pipeline_transformed_nodes


def main() -> None:
    setup_logging()

    try:

        print("started main")
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


        # === KNOWLEDGE EXTRACTOR PART ===

        # Load or create initial nodes
        filename = "./data/squirrel_image_test.pkl"
        initial_documents = get_initial_nodes(filename)

        # Initialise the pipeline
        pipeline = create_pipeline()

        # Load or create transformed nodes
        test_filename = "./data/squirrel-pipeline-image_embed_test.pkl"
        pipeline_transformed_nodes = create_transformed_nodes(
            initial_documents, test_filename, pipeline, embed_model
        )

        # Store nodes and relationships in Neo4j
        # uncoment later - already stored
        storage_manager.store_nodes(pipeline_transformed_nodes)

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise
    finally:
        storage_manager.close()


if __name__ == "__main__":
    main()