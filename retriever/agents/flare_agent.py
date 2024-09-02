from llama_index.core.query_engine import FLAREInstructQueryEngine

def process_question_with_flare(question, index, llm):
    """ Process a question using the FLARE query engine.
    
    Args:
        question (str): The question to process
        index (Index): The index to use for processing the question
        llm (LLM): The language model to use for processing the question
        
    Returns:
        dict: The response to the question    
    """

    query_engine = index.as_query_engine(multi_modal_llm=llm)

    flare_query_engine = FLAREInstructQueryEngine(
        query_engine=query_engine,
        max_iterations=3,
        verbose=True,
    )

    # Query with context
    enhanced_response = flare_query_engine.query(question)
    
    return enhanced_response
