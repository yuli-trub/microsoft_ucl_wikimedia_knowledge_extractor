import logging
from typing import List, Dict
from storage.storage_manager import StorageManager
from helper import log_duration
from llama_index.core import PromptTemplate
from typing import List
from llama_index.core.schema import NodeWithScore
from tqdm import tqdm
from llama_index.core import Settings
import json
from llama_index.core.schema import ImageNode


class GraphVectorRetriever:
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

    def vector_search(self, query_vector: List[float], top_k: int) -> List[str]:
        logging.info(f"Performing vector search in Qdrant with top_k={top_k}")
        search_results = self.storage_manager.vector_search(
            query_vector,
            top_k,
            node_type=None,
        )
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
            if parent_node and parent_node not in parent_nodes:
                parent_nodes.append(parent_node)
        # logging.info(f"Found parent node in Neo4j: {len(parent_nodes)}")
        return parent_nodes

    def retrieve_text_first(self, question, top_k=10):
        text_nodes = self.storage_manager.vector_search(
            question, top_k=top_k, node_type="text"
        )
        # logging.info(f"Text nodes: {text_nodes}")
        image_nodes = self.storage_manager.vector_search(
            question, top_k=top_k / 2, node_type="image"
        )
        # logging.info(f"Image nodes: {image_nodes}")

        combined_nodes = text_nodes + image_nodes
        return combined_nodes

    # fusion bit - https://docs.llamaindex.ai/en/stable/examples/low_level/fusion_retriever/
    # get queries
    def generate_queries(self, query_str: str) -> List[str]:
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

    #    run them
    def run_queries(
        self, queries: List[str], top_k: int
    ) -> Dict[str, List[NodeWithScore]]:
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
        k = 60.0
        text_weight = 1.5  # Boost text results
        fused_scores = {}
        text_to_node = {}

        for nodes_with_scores in results_dict.values():
            for rank, scored_point in enumerate(
                sorted(nodes_with_scores, key=lambda x: x.score, reverse=True)
            ):
                node_content = json.loads(scored_point.payload["_node_content"])
                point_id = node_content.get("id_", None)

                text_to_node[point_id] = scored_point
                if point_id not in fused_scores:
                    fused_scores[point_id] = 0.0
                boost = text_weight if scored_point.payload["type"] == "text" else 1.0
                fused_scores[point_id] += boost / (rank + k)

        logging.info(f"Fused scores: {fused_scores}")
        # Sort and rerank results
        reranked_results = sorted(
            fused_scores.items(), key=lambda x: x[1], reverse=True
        )
        logging.info(f"Reranked results: {len(reranked_results)}")

        reranked_nodes: List[NodeWithScore] = []
        for point_id, score in reranked_results[:similarity_top_k]:
            node_with_score = text_to_node[point_id]
            node_with_score.score = score
            reranked_nodes.append(node_with_score)

        logging.info(f"Reranked nodes: {(reranked_nodes)}")

        return reranked_nodes

    @log_duration
    def retrieve(self, query, top_k=10):
        query_vector = self.embed_model.get_query_embedding(query)
        search_results = self.retrieve_text_first(query_vector, top_k)
        llama_node_ids = self.get_llama_node_ids(search_results)
        nodes = self.retrieve_nodes_from_neo4j(llama_node_ids)
        parent_nodes = self.find_parent_nodes(llama_node_ids)
        return parent_nodes

    @log_duration
    def fusion_retrieve(self, query, top_k=10):
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
        images = [
            node.metadata["url"] for node in parent_nodes if isinstance(node, ImageNode)
        ]
        texts = [node.text for node in parent_nodes if hasattr(node, "text")]
        combined_text = " ".join(texts)
        combined_images = " ".join(images)
        logging.info(f"Texts from parent nodes: {texts}")
        logging.info(f"Images from parent nodes: {images}")
        return combined_text, combined_images
