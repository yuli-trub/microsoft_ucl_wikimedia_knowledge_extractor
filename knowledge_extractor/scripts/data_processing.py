import os
import logging

from scripts.helper import sanitise_filename, load_documents_from_file, save_documents_to_file
from scripts.wiki_crawler.searchinator import search_wiki
from scripts.llama_ingestionator.documentifier import process_page_into_doc_and_nodes
from scripts.llama_ingestionator.pipeline import run_pipeline

def get_initial_nodes(topic="test", num_pages=1, wiki_url = None) -> list:
    """Load initial nodes from a file or process a page to create them."""
    
    clean_topic = sanitise_filename(topic)
    filename = os.path.join(os.path.dirname(__file__), '..', 'data', f'{clean_topic}_initial')
    try:
        if os.path.exists(filename):
            try:
                documents = load_documents_from_file(filename)
                logging.info(f"Loaded {len(documents)} documents from {filename}")
            except Exception as e:
                if 'pickle data was truncated' in str(e):
                    logging.error(f"File {filename} is corrupted: {e}. Deleting the file and creating a new one.")
                    os.remove(filename)
                    documents = process_and_save_initial_documents(topic, num_pages, wiki_url, filename)
                else:
                    raise
        else:
            documents = process_and_save_initial_documents(topic, num_pages, wiki_url, filename)
    except Exception as e:
        logging.error(f"Failed to get initial nodes: {e}")
        raise
    return documents

def process_and_save_initial_documents(topic: str, num_pages: int, wiki_url: str, filename: str) -> list:
    """Process initial documents and save them to a file."""
    search_results = search_wiki(topic, wiki_url, num_pages)
    logging.info(f'search results: {search_results}')
    documents = []
    for title in search_results:
        results = process_page_into_doc_and_nodes(title) 
        documents.append(results)
    save_documents_to_file(documents, filename)
    logging.info(f"Processed and saved {len(documents)} documents")
    return documents



def create_transformed_nodes(
    documents: list, topic: str, pipeline, embed_model
) -> list:
    """Transform and save nodes using the pipeline."""

    clean_topic = sanitise_filename(topic)
    filename = os.path.join(os.path.dirname(__file__), '..', 'data', f'{clean_topic}_pipeline')

    try:
        if os.path.exists(filename):
            try:
                pipeline_transformed_nodes = load_documents_from_file(filename)
                logging.info(
                    f"Loaded {len(pipeline_transformed_nodes)} documents from {filename}"
                )
            except Exception as e:
                if 'pickle data was truncated' in str(e):
                    logging.error(f"File {filename} is corrupted: {e}. Deleting the file and creating a new one.")
                    os.remove(filename)
                    pipeline_transformed_nodes = process_and_save_transformed_documents(documents, filename, pipeline, embed_model)
                else:
                    raise
        else:
            pipeline_transformed_nodes = process_and_save_transformed_documents(documents, filename, pipeline, embed_model)
    except Exception as e:
        logging.error(f"Failed to create transformed nodes: {e}")
        raise
    return pipeline_transformed_nodes

def process_and_save_transformed_documents(documents: list, filename: str, pipeline, embed_model) -> list:
    """Process documents through the pipeline and save them to a file."""
    logging.info(f"Processing {len(documents)} documents")
    pipeline_transformed_nodes = []
    for doc in documents:
        logging.info(f"Processing document")
        transformed_nodes = run_pipeline(doc, pipeline, embed_model)
        logging.info(
            f"Processed and saved {len(transformed_nodes)} documents"
        )
        pipeline_transformed_nodes.append(transformed_nodes)
    save_documents_to_file(pipeline_transformed_nodes, filename)
    return pipeline_transformed_nodes
