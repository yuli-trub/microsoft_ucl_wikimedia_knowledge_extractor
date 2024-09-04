from knowledge_extractor.scripts.llama_ingestionator.transformator import EntityExtractorTransformation
from llama_index.core.schema import TextNode

def test_entity_extraction():
    text_node = TextNode(text="John works at OpenAI in San Francisco.", metadata={"type": "section", "title": "Test Node"})
    documents = [text_node]

    # Mocked entity extraction response
    class MockEntityExtractor(EntityExtractorTransformation):
        def openai_request(self, prompt, text=None, **kwargs):
            return {"choices": [{"message": {"content": '{"persons": [{"name": "John"}], "organizations": [{"name": "OpenAI"}], "locations": [{"name": "San Francisco"}], "dates": []}'}}]}

    entity_extractor = MockEntityExtractor()
    extracted_nodes = entity_extractor(documents)

    # check
    assert len(extracted_nodes) > len(documents), "No entity nodes were added"
