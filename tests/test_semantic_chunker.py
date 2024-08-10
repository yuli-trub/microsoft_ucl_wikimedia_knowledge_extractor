import unittest
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_ingestionator.transformator import SemanticChunker
from llama_index.core.schema import TextNode


class TestSemanticChunker(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Setup the OpenAI embedding model once for all tests
        cls.embed_model = OpenAIEmbedding()

    def setUp(self):
        # Initialize the SemanticChunker before each test
        self.semantic_chunker = SemanticChunker(embed_model=self.embed_model)

    def test_semantic_chunking(self):
        # Create a sample text node
        sample_text = (
            "This is a sample section text. It contains multiple sentences that should be "
            "chunked into semantically coherent parts. This text is used to test the "
            "semantic chunking functionality. Let's see how well it chunks this text. "
            "Semantic chunking should ensure that sentences with related meanings stay together."
        )
        sample_node = TextNode(
            text=sample_text,
            metadata={
                "id": "sample_section",
                "type": "section",
                "title": "Sample Section",
            },
        )

        # Process the sample node
        processed_nodes = self.semantic_chunker([sample_node])

        # Assertions to check the results
        self.assertGreater(
            len(processed_nodes), 1, "Semantic chunking did not create any chunks."
        )

        # Check the original node is included
        original_node = next(
            (
                node
                for node in processed_nodes
                if node.metadata["id"] == "sample_section"
            ),
            None,
        )
        self.assertIsNotNone(
            original_node, "Original node not found in processed nodes."
        )

        # Check for at least one chunk
        chunk_nodes = [
            node for node in processed_nodes if node.metadata["id"] != "sample_section"
        ]
        self.assertGreater(len(chunk_nodes), 0, "No chunk nodes created.")

        # Verify chunk metadata
        for chunk in chunk_nodes:
            self.assertIn(
                "parent_id",
                chunk.metadata,
                "Chunk node missing 'parent_id' in metadata.",
            )
            self.assertEqual(
                chunk.metadata["parent_id"],
                "sample_section",
                "Chunk node has incorrect 'parent_id'.",
            )

        # Print results for visual inspection
        print("Original Node:")
        print(f"ID: {sample_node.metadata['id']}")
        print(f"Content: {sample_node.text}\n")

        print("Processed Chunks:")
        for i, chunk in enumerate(chunk_nodes):
            print(f"Chunk {i+1}:")
            print(f"ID: {chunk.metadata['id']}")
            print(f"Parent ID: {chunk.metadata['parent_id']}")
            print(f"Content: {chunk.text}\n")


if __name__ == "__main__":
    unittest.main()
