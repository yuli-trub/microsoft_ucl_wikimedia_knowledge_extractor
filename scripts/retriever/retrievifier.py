import logging
from typing import List, Dict
from storage.storage_manager import StorageManager
from helper import log_duration


class Retriever:
    def __init__(self, storage_manager: StorageManager, embed_model):
        self.storage_manager = storage_manager
        self.embed_model = embed_model

    def vector_search(self, query_vector: List[float], top_k: int) -> List[str]:
        logging.info(f"Performing vector search in Qdrant with top_k={top_k}")
        search_results = self.storage_manager.vector_search(query_vector, top_k)
        logging.info(f"Vector search results: {search_results}")
        return search_results

    def get_llama_node_ids(self, search_results):
        llama_node_ids = [result.payload["llama_node_id"] for result in search_results]
        logging.info(f"Llama Node IDs: {llama_node_ids}")
        return llama_node_ids

    def retrieve_nodes_from_neo4j(self, llama_node_ids: List[str]) -> List[Dict]:
        # logging.info(
        #     f"Retrieving nodes from Neo4j using llama_node_ids: {llama_node_ids}"
        # )
        nodes = []
        for node_id in llama_node_ids:
            # logging.info(f"Retrieving node from Neo4j with Llama Node ID: {node_id}")
            node = self.storage_manager.neo4j_client.get_node_by_llama_id(node_id)
            if node:
                nodes.append(node)
                logging.info(f"Retrieved node from Neo4j: {node}")
        return nodes

    def find_parent_nodes(self, llama_node_ids: List[str]) -> List[Dict]:
        logging.info(
            f"Finding parent nodes in Neo4j for no Llama Node IDs: {len(llama_node_ids)}"
        )
        parent_nodes = []
        for node_id in llama_node_ids:
            parent_node = self.storage_manager.neo4j_client.get_parent_node(node_id)
            # logging.info(f"Parent node in Neo4j: {parent_node}")
            if parent_node:
                parent_nodes.append(parent_node)
        # logging.info(f"Found parent node in Neo4j: {len(parent_nodes)}")
        return parent_nodes

    def retrieve_text_first(self, question, top_k=10):
        text_nodes = self.storage_manager.vector_search(
            question, top_k=top_k, node_type="text"
        )
        # logging.info(f"Text nodes: {text_nodes}")
        image_nodes = self.storage_manager.vector_search(
            question, top_k=top_k, node_type="image"
        )
        # logging.info(f"Image nodes: {image_nodes}")

        combined_nodes = text_nodes + image_nodes
        return combined_nodes

    @log_duration
    def retrieve(self, query, top_k=10):
        query_vector = self.embed_model.get_query_embedding(query)
        search_results = self.retrieve_text_first(query_vector, top_k)
        llama_node_ids = self.get_llama_node_ids(search_results)
        # nodes = self.retrieve_nodes_from_neo4j(llama_node_ids)
        parent_nodes = self.find_parent_nodes(llama_node_ids)
        return parent_nodes
