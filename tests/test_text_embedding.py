from knowledge_extractor.scripts.llama_ingestionator.transformator import EmbeddingTransformation
from llama_index.core.schema import TextNode

def test_embedding_transformation():
    # Mocked TextNode data
    text_node = TextNode(text="Sample text for embedding", metadata={"needs_embedding": True, "title": "Test Node"})
    documents = [text_node]

    # Mocked embedding model
    class MockEmbedModel:
        def get_text_embedding(self, text):
            return [0.1, 0.2, 0.3]  

    embedding_transformation = EmbeddingTransformation()
    transformed_docs = embedding_transformation(documents, text_embed_model=MockEmbedModel())

    # Check that embedding was added
    assert transformed_docs[0].embedding == [0.1, 0.2, 0.3], "Embedding was not added correctly"
