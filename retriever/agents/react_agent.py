from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata

def process_question_with_react(question, retriever, llm):
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

    # Initialisnig ReAct agent with the LLM and my tools
    react_agent = ReActAgent.from_tools(
        query_engine_tools,
        llm=llm,
        verbose=True,
    )

    # process the question
    enhanced_response = react_agent.chat(question)
    
    return enhanced_response
