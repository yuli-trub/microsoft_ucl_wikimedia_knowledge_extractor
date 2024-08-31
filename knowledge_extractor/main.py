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

import re



def get_initial_nodes(topic="test", num_pages=1, wiki_url = None) -> list:
    """Load initial nodes from a file or process a page to create them."""
    
    clean_topic = sanitise_filename(topic)
    # to change later after cleaning
    filename = f'./data/{clean_topic}_initial_test'

    try:
        if os.path.exists(filename):
            documents = load_documents_from_file(filename)
            logging.info(f"Loaded {len(documents)} documents from {filename}")
        else:
            search_results = search_wiki(topic, wiki_url, num_pages)[0]
            logging.info(f'search results: {search_results}')
            documents=[]
            for title in search_results:
                results = process_page_into_doc_and_nodes(title) 
                documents.append(results)
            save_documents_to_file(documents, filename)
            logging.info(f"Processed and saved {len(documents)} documents")
    except Exception as e:
        logging.error(f"Failed to get initial nodes: {e}")
        raise
    return documents


def create_transformed_nodes(
    documents: list, topic: str, pipeline, embed_model
) -> list:
    """Transform and save nodes using the pipeline."""

    clean_topic = sanitise_filename(topic)
    filename = f'./data/{clean_topic}_pipeline'
    try:
        if os.path.exists(filename):
            pipeline_transformed_nodes = load_documents_from_file(filename)
            logging.info(
                f"Loaded {len(pipeline_transformed_nodes)} documents from {filename}"
            )
        else:
            logging.info(f"Processing {len(documents)} documents")
            pipeline_transformed_nodes =[]
            for doc in documents:
                logging.info(f"Processing {len(doc)} documents")
                transformaed_nodes = run_pipeline(doc, pipeline, embed_model)
                logging.info(
                    f"Processed and saved {len(transformaed_nodes)} documents"
                )
                pipeline_transformed_nodes.append(transformaed_nodes)
            save_documents_to_file(pipeline_transformed_nodes, filename)
    except Exception as e:
        logging.error(f"Failed to create transformed nodes: {e}")
        raise
    return pipeline_transformed_nodes


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

        # # Initialise the pipeline
        # pipeline = create_pipeline()

        # # Load or create transformed nodes
        # # test_filename = "./data/squirrel-pipeline-image_embed_test.pkl"
        # pipeline_transformed_nodes = create_transformed_nodes(
        #     initial_documents, env_vars['DOMAIN_TOPIC'], pipeline, embed_model
        # )

        # # Store nodes and relationships in Neo4j
        # # uncoment later - already stored
        # for page in pipeline_transformed_nodes:
        #     storage_manager.store_nodes(page)

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise
    finally:
        storage_manager.close()


if __name__ == "__main__":
    main()