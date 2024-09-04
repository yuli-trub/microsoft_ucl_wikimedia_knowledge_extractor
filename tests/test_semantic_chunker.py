from knowledge_extractor.scripts.llama_ingestionator.transformator import SemanticChunkingTransformation


def test_transformation_pipeline():

    text_content = """
    Artificial intelligence (AI) refers to the simulation of human intelligence in machines that are designed to think and act like humans. 
    The term may also be applied to any machine that exhibits traits associated with a human mind such as learning and problem-solving. 
    The ideal characteristic of artificial intelligence is its ability to rationalize and take actions that have the best chance of achieving a specific goal.
    
    AI is being used across various industries, including healthcare, finance, and robotics, to improve efficiency and reduce the need for human labor. 
    In healthcare, AI algorithms can assist doctors in diagnosing diseases with greater accuracy. In finance, AI is used for fraud detection, algorithmic trading, 
    and personal financial planning. Robotics leverages AI for tasks such as navigation and object manipulation.

    However, there are concerns about AI's impact on employment and the ethical implications of delegating decision-making to machines. 
    AI systems can sometimes reflect biases in the data they were trained on, which can lead to unintended and potentially harmful consequences. 
    Researchers continue to work on addressing these challenges while advancing the technology's capabilities.
    """

    semantic_chunking = SemanticChunkingTransformation()
    chunks = semantic_chunking(text_content)
    
    # Check that chunking works correctly
    assert isinstance(chunks, list)
    assert len(chunks) > 0
    assert chunks[0] in text_content