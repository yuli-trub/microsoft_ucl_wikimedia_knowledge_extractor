from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata

def process_question_with_react(question, retriever, llm):
    """ Process a question using the ReAct agent.
    
    Args:
        question (str): The question to process
        retriever (Retriever): The retriever to use for processing the question
        llm (LLM): The language model to use for processing the question

    Returns:
        dict: The response to the question
    """
    # Set up the query tools for the ReAct agent
    query_engine_tools = [
        QueryEngineTool(
            query_engine=retriever,
            metadata=ToolMetadata(
                name="custom_retriever",
                description="Custom retriever providing context and information based on the question from the domain knowledge base to get enhanced responses."
            ),
        )
    ]

    # Initialising ReAct agent with the LLM and my tool
    react_agent = ReActAgent.from_tools(
        query_engine_tools,
        llm=llm,
        verbose=True,
    )

    # process the question
    enhanced_response = react_agent.chat(question)
    
    return enhanced_response
