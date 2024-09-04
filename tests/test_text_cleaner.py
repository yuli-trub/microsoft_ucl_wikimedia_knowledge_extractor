from knowledge_extractor.scripts.llama_ingestionator.transformator import TextCleaner
from llama_index.core.schema import TextNode

def test_text_cleaner():
    # Mocked TextNode with special characters
    text_node = TextNode(text="This! is a, test with: special characters.", metadata={"type": "section"})
    nodes = [text_node]

    text_cleaner = TextCleaner()
    cleaned_nodes = text_cleaner(nodes)

    # Check that special characters are removed
    assert cleaned_nodes[0].text == "This is a test with special characters", "Text cleaning failed"
