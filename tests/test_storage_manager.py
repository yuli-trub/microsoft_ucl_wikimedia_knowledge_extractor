import unittest
from unittest.mock import patch, MagicMock
from knowledge_extractor.scripts.storage.storage_manager import StorageManager
from llama_index.core.schema import Document, TextNode, ImageNode

class TestStorageManager(unittest.TestCase):

    @patch("knowledge_extractor.scripts.storage.storage_manager.Neo4jClient")
    @patch("knowledge_extractor.scripts.storage.storage_manager.setup_qdrant_client")
    def setUp(self, mock_setup_qdrant_client, mock_neo4j_client):
        # Mock Neo4jClient and Qdrant
        self.mock_neo4j_client = mock_neo4j_client.return_value
        mock_setup_qdrant_client.return_value = ("mock_qdrant_client", "mock_text_store", "mock_image_store", None, None)

        # Set up the StorageManager with mock configurations
        neo4j_config = {"uri": "bolt://localhost:7687", "user": "neo4j", "password": "password"}
        qdrant_config = {"host": "localhost", "port": 6333}
        self.storage_manager = StorageManager(neo4j_config, qdrant_config)

    def test_store_nodes_and_relationships(self):
        # Mock nodes
        document_node = Document(node_id="doc_1", metadata={"title": "Test Doc"})
        text_node = TextNode(node_id="text_1", text="Test Text", metadata={"title": "Test Text"})
        image_node = ImageNode(node_id="image_1", metadata={"title": "Test Image", "url": "http://example.com/image.jpg"})
        
        nodes = [document_node, text_node, image_node]

        # Mock Neo4j operations
        self.mock_neo4j_client.create_document_node.return_value = "neo4j_doc_1"
        self.mock_neo4j_client.create_text_node.return_value = "neo4j_text_1"
        self.mock_neo4j_client.create_image_node.return_value = "neo4j_image_1"

        # Run the store_nodes_and_relationships method
        id_map = self.storage_manager.store_nodes_and_relationships(nodes)

        # Assertions to ensure that the nodes were stored correctly in Neo4j
        self.mock_neo4j_client.create_document_node.assert_called_once_with(document_node)
        self.mock_neo4j_client.create_text_node.assert_called_once_with(text_node)
        self.mock_neo4j_client.create_image_node.assert_called_once_with(image_node)
        
        # Check that the id map is correctly generated
        self.assertEqual(id_map, {"doc_1": "neo4j_doc_1", "text_1": "neo4j_text_1", "image_1": "neo4j_image_1"})

    @patch("knowledge_extractor.scripts.storage.storage_manager.add_node_to_qdrant")
    def test_add_nodes_to_qdrant(self, mock_add_node_to_qdrant):
        # Mock nodes
        document_node = Document(node_id="doc_1", metadata={"title": "Test Doc"}, embedding=[0.1, 0.2, 0.3])
        text_node = TextNode(node_id="text_1", text="Test Text", embedding=[0.1, 0.2, 0.3])
        image_node = ImageNode(node_id="image_1", metadata={"title": "Test Image", "url": "http://example.com/image.jpg"}, embedding=[0.1, 0.2, 0.3])
        
        nodes = [document_node, text_node, image_node]

        # ID map (mapping Llama node IDs to Neo4j node IDs)
        id_map = {"doc_1": "neo4j_doc_1", "text_1": "neo4j_text_1", "image_1": "neo4j_image_1"}

        # Run add_nodes_to_qdrant method
        self.storage_manager.add_nodes_to_qdrant(nodes, id_map)

        # Ensure add_node_to_qdrant is called correctly for nodes with embeddings
        mock_add_node_to_qdrant.assert_any_call("mock_text_store", text_node, "neo4j_text_1", "text_1")
        mock_add_node_to_qdrant.assert_any_call("mock_image_store", image_node, "neo4j_image_1", "image_1")

    @patch("scripts.storage.storage_manager.add_node_to_qdrant")
    def test_store_nodes(self, mock_add_node_to_qdrant):
        # Mock nodes
        document_node = Document(node_id="doc_1", metadata={"title": "Test Doc"}, embedding=[0.1, 0.2, 0.3])
        text_node = TextNode(node_id="text_1", text="Test Text", embedding=[0.1, 0.2, 0.3])
        image_node = ImageNode(node_id="image_1", metadata={"title": "Test Image", "url": "http://example.com/image.jpg"}, embedding=[0.1, 0.2, 0.3])
        
        nodes = [document_node, text_node, image_node]

        # Mock the Neo4j client return values
        self.mock_neo4j_client.create_document_node.return_value = "neo4j_doc_1"
        self.mock_neo4j_client.create_text_node.return_value = "neo4j_text_1"
        self.mock_neo4j_client.create_image_node.return_value = "neo4j_image_1"

        # Run store_nodes
        self.storage_manager.store_nodes(nodes)

        # Check if the nodes are correctly stored in Neo4j
        self.mock_neo4j_client.create_document_node.assert_called_once_with(document_node)
        self.mock_neo4j_client.create_text_node.assert_called_once_with(text_node)
        self.mock_neo4j_client.create_image_node.assert_called_once_with(image_node)

        # Ensure add_node_to_qdrant is called for nodes with embeddings
        mock_add_node_to_qdrant.assert_any_call("mock_text_store", text_node, "neo4j_text_1", "text_1")
        mock_add_node_to_qdrant.assert_any_call("mock_image_store", image_node, "neo4j_image_1", "image_1")

    def test_close(self):
        # Call close method
        self.storage_manager.close()

        # Ensure Neo4j client close method is called
        self.mock_neo4j_client.close.assert_called_once()

if __name__ == "__main__":
    unittest.main()

