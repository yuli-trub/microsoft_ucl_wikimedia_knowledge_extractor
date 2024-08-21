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

import gradio as gr


# import storage_manager
from scripts.storage.storage_manager import StorageManager

# import retriever
from scripts.retriever.retrievifier import GraphVectorRetriever

# import agent and FLARE
from agents import process_question_with_flare, process_question_with_react

def process_question(question):
    try:

        print("started retriever main")
        # Load environment variables
        env_vars = get_env_vars()
        logging.info(env_vars)

        # Neo4j and Qdrant configurations
        neo4j_config = get_neo4j_config(env_vars)
        qdrant_config = get_qdrant_config(env_vars)

        # Set up embedding model
        embed_model = initialise_embed_model(env_vars)

        # Set up LLM
        llm = initialise_llm(env_vars)

        # Initialise StorageManager
        storage_manager = StorageManager(neo4j_config, qdrant_config)


        # Build index
        index = storage_manager.build_index()

        # === RETRIEVAL PART ===
        # Retrieval stage
        retriever = GraphVectorRetriever(storage_manager, embed_model)

        # retrieve original nodes 
        parent_nodes = retriever.fusion_retrieve(question)
        logging.info(
            f"Retrieved parent nodes: {[node.metadata['type'] for node in parent_nodes]}"
        )

        # Get context from retrieved nodes
        combined_text, combined_images = retriever.get_context_from_retrived_nodes(
            parent_nodes
        )

        # === QUERY PART ===
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
            "- If no context was provided, please state that the context wasn't provided to answer the question"
            f"Question: {question}\n"
            "Answer:"
        )
        llm_input = f"Question: {question}\n\nAnswer:"

        # process with FLARE
        # flare_enhanced_response = process_question_with_flare(llm_rag_input, index, llm)

        # process with ReAct
        enhanced_response = process_question_with_react(llm_rag_input, retriever, llm)
        standard_response = llm.complete(llm_input)

        # Log and print both responses for comparison
        logging.info(f"Enhanced Response: {enhanced_response}")
        logging.info(f"Standard Response: {standard_response}")

        return str(enhanced_response), str(standard_response)

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise
    finally:
        storage_manager.close()


def main() -> None:
    setup_logging()

    with gr.Blocks() as demo:
        gr.Markdown("# Enhanced vs Standard Response")
        gr.Markdown("Ask a question and see the difference between an enhanced response (using retrieved context) and a standard response.")
        
        # Full-width question input
        question = gr.Textbox(label="Enter your question", placeholder="Type your question here...", elem_id="question_box")

        # Example questions below
        with gr.Row():
            gr.Examples(examples=[["What are the main characteristics of squirrels and what do they like to eat?"]], inputs=question)

        # Two columns for enhanced and standard responses to compare
        with gr.Row():
            with gr.Column():
                gr.Markdown("## Enhanced Response:")
                enhanced_response = gr.Markdown(label="Enhanced Response")
            with gr.Column():
                gr.Markdown("## Standard Response:")
                standard_response = gr.Markdown(label="Standard Response")

        # Submit button to process the question
        question.submit(process_question, inputs=question, outputs=[enhanced_response, standard_response])

    demo.launch(server_name="0.0.0.0", server_port=5000)

if __name__ == "__main__":
    main()
