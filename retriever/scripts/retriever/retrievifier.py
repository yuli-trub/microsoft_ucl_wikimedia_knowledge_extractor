import logging
from typing import List, Dict
from scripts.storage.storage_manager import StorageManager
from scripts.helper import log_duration
from llama_index.core import PromptTemplate
from typing import List
from llama_index.core.schema import NodeWithScore
from tqdm import tqdm
from llama_index.core import Settings
import json
from llama_index.core.schema import ImageNode


class GraphVectorRetriever:
    """Custom retriever class for retrieving nodes from Qdrant and Neo4j."""
    def __init__(
        self,
        storage_manager: StorageManager,
        embed_model,
        num_queries=4,
    ):
        self.storage_manager = storage_manager
        self.embed_model = embed_model
        self.num_queries = num_queries
        self.llm = Settings.llm

    # def vector_search(self, query_vector: List[float], top_k: int) -> List[str]:
    #     logging.info(f"Performing vector search in Qdrant with top_k={top_k}")
    #     search_results = self.storage_manager.vector_search(
    #         query_vector,
    #         top_k,
    #         node_type=None,
    #     )
    #     return search_results

    def get_llama_node_ids(self, search_results):
        """Get Llama Node IDs from search results from the payload
        
        Args:
            search_results: reranked vector search results with payload

        Returns:
            List[str]: List of Llama Node IDs
        """
        llama_node_ids = [result.payload["llama_node_id"] for result in search_results]
        logging.info(f"Llama Node IDs: {llama_node_ids}")
        return llama_node_ids

    def retrieve_nodes_from_neo4j(self, llama_node_ids: List[str]) -> List[Dict]:
        """ Retrieve nodes from Neo4j using Llama Node IDs
        
        Args:
            llama_node_ids (List[str]): List of Llama Node IDs
            
        Returns:
            List[Dict]: List of nodes retrieved from Neo4j
        """
        nodes = []
        for node_id in llama_node_ids:
            node = self.storage_manager.neo4j_client.get_node_by_llama_id(node_id)
            if node:
                nodes.append(node)
        logging.info(f"Retrieved {len(nodes)} nodes from Neo4j")
        return nodes

    def find_parent_nodes(self, llama_node_ids: List[str]) -> List[Dict]:
        """Retrieve parent nodes from Neo4j for the given Llama Node IDs - the original data from Wikimedia
        
        Args:
            llama_node_ids (List[str]): List of Llama Node IDs
        
        Returns:
            List[Dict]: List of parent nodes retrieved from Neo4j
        """
        logging.info(
            f"Finding parent nodes in Neo4j for Llama Node IDs"
        )
        parent_nodes = []
        for node_id in llama_node_ids:
            parent_node = self.storage_manager.neo4j_client.get_parent_node(node_id)
            if parent_node and parent_node not in parent_nodes:
                parent_nodes.append(parent_node)
        logging.info(f"Found {len(parent_nodes)} original parent nodes in Neo4j")
        return parent_nodes

    def retrieve_text_first(self, question, top_k=10):
        """Retrieve text and image nodes from Qdrant with text being the priority
        
        Args:
            question (str): The input query
            top_k (int): The number of results to retrieve
            
        Returns:
            List[NodeWithScore]: List of nodes retrieved from Qdrant
        """

        text_nodes = self.storage_manager.vector_search(
            question, top_k=top_k, node_type="text"
        )
        image_nodes = self.storage_manager.vector_search(
            question, top_k=top_k / 2, node_type="image"
        )

        combined_nodes = text_nodes + image_nodes
        return combined_nodes

    # fusion bit - https://docs.llamaindex.ai/en/stable/examples/low_level/fusion_retriever/

    def generate_queries(self, query_str: str) -> List[str]:
        """Fusion retrival: Generate multiple search queries based on a single input query
        
        Args:
            query_str (str): The input query

        Returns:
            List[str]: List of generated search queries
        """
        query_gen_prompt_str = (
            "You are a helpful assistant that generates multiple search queries based on a "
            "single input query. Generate {num_queries} search queries, one on each line, "
            "related to the following input query:\n"
            "Query: {query}\n"
            "Queries:\n"
        )
        query_gen_prompt = PromptTemplate(query_gen_prompt_str)
        fmt_prompt = query_gen_prompt.format(
            num_queries=self.num_queries, query=query_str
        )
        response = self.llm.complete(fmt_prompt)
        queries = response.text.strip().split("\n")
        for query in queries:
            logging.info(f"Generated query: {query}")
        # return the list including the original query as well
        return [query_str] + queries


    def run_queries(
        self, queries: List[str], top_k: int
    ) -> Dict[str, List[NodeWithScore]]:
        """Fusion retrieval: Run multiple search queries and retrieve results from Qdrant
        
        Args:
            queries (List[str]): List of search queries

        Returns:
            Dict[str, List[NodeWithScore]]: Dictionary of search results for each query
        """
        results_dict = {}
        for query in tqdm(queries, desc="Running Queries"):
            query_vector = self.embed_model.get_query_embedding(query)
            search_results = self.retrieve_text_first(query_vector, top_k)
            logging.info(f"Search results for query: {query}, {len(search_results)}")
            results_dict[query] = search_results

        return results_dict

    # rerank results and boost text results
    def fuse_results(
        self, results_dict: Dict[str, List[NodeWithScore]], similarity_top_k: int = 10
    ) -> List[NodeWithScore]:
        """Reranker: Fuse and rerank search results
        
        Args:
            results_dict (Dict[str, List[NodeWithScore]]): Dictionary of search results for each query
            similarity_top_k (int): The number of results to return after reranking

        Returns:
            List[NodeWithScore]: List of reranked nodes
        """
        k = 60.0 # Normalisation factor
        text_weight = 1.5  # Boost text results
        fused_scores = {}
        text_to_node = {}

        for nodes_with_scores in results_dict.values():
            for rank, scored_point in enumerate(
                sorted(nodes_with_scores, key=lambda x: x.score, reverse=True)
            ):
                # Extract point ID
                node_content = json.loads(scored_point.payload["_node_content"])
                point_id = node_content.get("id_", None)
                # Store nodes for reranking
                text_to_node[point_id] = scored_point
                if point_id not in fused_scores:
                    fused_scores[point_id] = 0.0
                # Boost text results
                boost = text_weight if scored_point.payload["type"] == "text" else 1.0
                fused_scores[point_id] += boost / (rank + k)

        # Sort reranked results
        reranked_results = sorted(
            fused_scores.items(), key=lambda x: x[1], reverse=True
        )

        # Extract top k results
        reranked_nodes: List[NodeWithScore] = []
        for point_id, score in reranked_results[:similarity_top_k]:
            node_with_score = text_to_node[point_id]
            node_with_score.score = score
            reranked_nodes.append(node_with_score)

        return reranked_nodes

    # @log_duration
    # def retrieve(self, query, top_k=10):
    #     """Retrieve nodes for context from Neo4j and Qdrant for a given query"""
    #     query_vector = self.embed_model.get_query_embedding(query)
    #     search_results = self.retrieve_text_first(query_vector, top_k)
    #     llama_node_ids = self.get_llama_node_ids(search_results)
    #     nodes = self.retrieve_nodes_from_neo4j(llama_node_ids)
    #     parent_nodes = self.find_parent_nodes(llama_node_ids)
    #     return parent_nodes

    @log_duration
    def fusion_retrieve(self, query, top_k=10):
        """Retrieve nodes for context from Neo4j and Qdrant for a given query using fusion retrieval and reranker
        
        Args:
            query (str): The input query
            top_k (int): The number of results to retrieve

        Returns:
            List[Dict]: List of parent nodes with original data from Wikimedia retrieved from Neo4j
        """
        # generate extra queries
        queries = self.generate_queries(query)

        # vector search for each query
        results_dict = self.run_queries(queries, top_k)

        # Fuse results and rerank
        fused_results = self.fuse_results(results_dict, similarity_top_k=top_k)

        # Retrieve nodes from Neo4j
        llama_node_ids = self.get_llama_node_ids(fused_results)

        parent_nodes = self.find_parent_nodes(llama_node_ids)

        return parent_nodes

    def get_context_from_retrived_nodes(self, parent_nodes):
        """Get combined text and images from the retrieved parent nodes to create context
        
        Args:
            parent_nodes: List of parent nodes retrieved from Neo4j

        Returns:
            Tuple[str, str]: Combined text and images from the parent nodes
        """
        images = [
            node.metadata["url"] for node in parent_nodes if isinstance(node, ImageNode)
        ]
        texts = [node.text for node in parent_nodes if hasattr(node, "text")]
        combined_text = " ".join(texts)
        combined_images = " ".join(images)
        logging.info(f"Texts from parent nodes: {texts}")
        logging.info(f"Images from parent nodes: {images}")
        return combined_text, combined_images
