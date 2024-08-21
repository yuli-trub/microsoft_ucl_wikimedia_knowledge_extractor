from llama_index.core.query_engine import FLAREInstructQueryEngine

def process_question_with_flare(question, index, llm):

    query_engine = index.as_query_engine(multi_modal_llm=llm)

    flare_query_engine = FLAREInstructQueryEngine(
        query_engine=query_engine,
        max_iterations=3,
        verbose=True,
    )

    # Query with context
    enhanced_response = flare_query_engine.query(question)
    
    return enhanced_response
