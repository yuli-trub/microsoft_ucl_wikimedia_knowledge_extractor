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

# import retriever
from scripts.retriever.retrievifier import GraphVectorRetriever

# import FLARE
from llama_index.core.query_engine import FLAREInstructQueryEngine


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
        # Load environment variables
        env_vars = get_env_vars()

        # Neo4j and Qdrant configurations
        neo4j_config = get_neo4j_config(env_vars)
        qdrant_config = get_qdrant_config(env_vars)

        # Initialise StorageManager
        storage_manager = StorageManager(neo4j_config, qdrant_config)

        # Set up embedding model
        embed_model = initialise_embed_model(env_vars)

        # Set up LLM
        llm = initialise_llm(env_vars)

        # === KNOWLEDGE EXTRACTOR PART ===

        # Load or create initial nodes
        filename = "squirrel_image_test.pkl"
        initial_documents = get_initial_nodes(filename)

        # Initialise the pipeline
        pipeline = create_pipeline()

        # Load or create transformed nodes
        test_filename = "squirrel-pipeline-image_embed_test.pkl"
        pipeline_transformed_nodes = create_transformed_nodes(
            initial_documents, test_filename, pipeline, embed_model
        )

        # Store nodes and relationships in Neo4j
        # uncoment later - already stored
        # storage_manager.store_nodes(pipeline_transformed_nodes)

        # Build index
        index = storage_manager.build_index()

        # === RETRIEVAL PART ===
        # Retrieval stage
        retriever = GraphVectorRetriever(storage_manager, embed_model)

        # Example question
        question = (
            "What is the main characteristics of squirrel and what do they like to eat?"
        )

        parent_nodes = retriever.fusion_retrieve(question)
        logging.info(
            f"Retrieved parent nodes: {[node.metadata['type'] for node in parent_nodes]}"
        )

        # Get context from retrieved nodes
        combined_text, combined_images = retriever.get_context_from_retrived_nodes(
            parent_nodes
        )

        # === FLARE QUERY ENGINE - QUERY PART===
        # LLM input
        llm_rag_input = (
            "You are provided with context information retrieved from various sources. "
            "Please use this information to answer the following question thoroughly and concisely.\n\n"
            "Context (Textual Information):\n"
            f"{combined_text}\n"
            "\nRelevant Images (URLs) (if applicable):\n"
            f"{combined_images}\n"
            "\nInstructions:\n"
            "- Base your answer primarily on the textual context provided.\n"
            "- Use relevant details from the images only if they add value to the answer.\n"
            "- Structure your response using headings and bullet points for clarity.\n"
            "- Avoid repeating information.\n"
            "- Ensure the answer is informative and directly addresses the question.\n\n"
            f"Question: {question}\n"
            "Answer:"
        )
        llm_input = f"Question: {question}\n\nAnswer:"

        query_engine = index.as_query_engine(multi_modal_llm=llm)

        flare_query_engine = FLAREInstructQueryEngine(
            query_engine=query_engine,
            max_iterations=7,
            verbose=True,
        )

        # Query with context
        standard_response = llm.complete(llm_input)
        enhanced_response = flare_query_engine.query(llm_rag_input)

        # Log and print both responses for comparison
        logging.info(f"Enhanced Response: {enhanced_response}")
        logging.info(f"Standard Response: {standard_response}")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise
    finally:
        storage_manager.close()


if __name__ == "__main__":
    main()
