import os
import requests
from llama_index.core.schema import TransformComponent
import re
from llama_index.core.schema import TextNode
from llama_index.core.node_parser import SemanticSplitterNodeParser
import logging
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core import Settings
from pydantic import Field
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential_jitter,
    retry_if_exception_type,
)
import time
from helper import load_env


logging.basicConfig(
    level=logging.INFO,
    filename="pipeline.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
)


env_vars = load_env(
    "AZURE_OPENAI_API_KEY",
    "OPENAI_ENDPOINT",
    "GPT4O_DEPLOYMENT_ID",
    "GPT4O_API_VERSION",
    "GPT4_ENDPOINT",
    "EMBEDDING_DEPLOYMENT_ID",
    "EMBEDDING_API_VERSION",
)

AZURE_OPENAI_API_KEY = env_vars["AZURE_OPENAI_API_KEY"]
OPENAI_ENDPOINT = env_vars["OPENAI_ENDPOINT"]
GPT4O_DEPLOYMENT_ID = env_vars["GPT4O_DEPLOYMENT_ID"]
GPT4O_API_VERSION = env_vars["GPT4O_API_VERSION"]
GPT4_ENDPOINT = env_vars["GPT4_ENDPOINT"]
EMBEDDING_DEPLOYMENT_ID = env_vars["EMBEDDING_DEPLOYMENT_ID"]
EMBEDDING_API_VERSION = env_vars["EMBEDDING_API_VERSION"]

headers = {
    "Content-Type": "application/json",
    "api-key": AZURE_OPENAI_API_KEY,
}


# Configure global settings
Settings.embed_model = AzureOpenAIEmbedding(
    model="text-embedding-ada-002",
    deployment_name=EMBEDDING_DEPLOYMENT_ID,
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=OPENAI_ENDPOINT,
    api_version=EMBEDDING_API_VERSION,
)


Settings.text_splitter = SemanticSplitterNodeParser(
    buffer_size=1, breakpoint_percentile_threshold=80, embed_model=Settings.embed_model
)


# text cleaner from llamaindex
class TextCleaner(TransformComponent):
    def __call__(self, nodes, **kwargs):
        print(f"Processing {len(nodes)} nodes")
        for node in nodes:
            if node.metadata.get("type") in ["section", "subsection"]:
                print(f"Processing node ID: {node.metadata['id']}")
                print(f"Original text: {node.text[:150]}...")
                node.text = re.sub(r"[^0-9A-Za-z ]", "", node.text)
                print(f"Cleaned text: {node.text[:150]}...")

        return nodes


class OpenAIBaseTransformation(TransformComponent):
    @retry(
        wait=wait_exponential_jitter(initial=1, max=60),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type(requests.exceptions.RequestException),
    )
    def openai_request(self, prompt, text, function=None):
        payload = {
            "messages": [
                {"role": "system", "content": "You are an AI assistant."},
                {"role": "user", "content": f"{prompt} {text}"},
            ],
            "temperature": 0.7,
            "top_p": 0.95,
            "max_tokens": 800,
        }
        if function:
            payload["functions"] = [function]
            payload["function_call"] = {"name": function["name"]}

        try:
            response = requests.post(GPT4_ENDPOINT, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            if response.status_code == 429:
                logging.warning("Rate limit exceeded. Retrying...")
                time.sleep(10)
            raise

    def get_response(self, response):
        try:
            token_usage = response["usage"]["total_tokens"]
            logging.info(f"Tokens used: {token_usage}")
            cost = self.calculate_cost(token_usage)
            logging.info(f"Estimated cost: ${cost:.2f}")
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            logging.error("Failed to retrieve summary from response.")
            return "Transformation failed or unclear"

    def calculate_cost(self, tokens):
        return tokens * 0.00002


class SemanticChunkingTransformation(TransformComponent):
    def __call__(self, documents, **kwargs):
        transformed_nodes = []
        splitter = Settings.text_splitter  # Use global setting

        for node in documents:
            if node.metadata.get("type") in ["section", "subsection"]:
                chunks = splitter.get_nodes_from_documents([node])
                for idx, chunk in enumerate(chunks):
                    chunk.metadata["id"] = f"{node.metadata['id']}_chunk_{idx}"
                    chunk.metadata["parent_id"] = node.metadata["id"]
                    chunk.metadata["type"] = "chunk"
                transformed_nodes.extend(chunks)
            else:
                transformed_nodes.append(node)
        return documents + transformed_nodes


class EntityExtractorTransformation(OpenAIBaseTransformation):
    # TODO: figure out the format to return
    function: dict = Field(
        {
            "name": "extract_entities",
            "description": "Extract entities from text and return as structured JSON",
            "parameters": {
                "type": "object",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "persons": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "type": {"type": "string", "default": "person"},
                                    "name": {"type": "string"},
                                    "context": {"type": "string"},
                                },
                                "required": ["type", "name", "context"],
                            },
                        },
                        "organizations": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "type": {
                                        "type": "string",
                                        "default": "organization",
                                    },
                                    "name": {"type": "string"},
                                    "context": {"type": "string"},
                                },
                                "required": ["type", "name", "context"],
                            },
                        },
                        "locations": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "type": {"type": "string", "default": "location"},
                                    "name": {"type": "string"},
                                    "context": {"type": "string"},
                                },
                                "required": ["type", "name", "context"],
                            },
                        },
                        "dates": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "type": {"type": "string", "default": "date"},
                                    "name": {"type": "string"},
                                    "context": {"type": "string"},
                                },
                                "required": ["type", "name", "context"],
                            },
                        },
                    },
                    "required": ["persons", "organizations", "locations", "dates"],
                },
            },
        },
        alias="extract_entities",
    )

    def __call__(self, documents, **kwargs):
        entities_nodes = []
        prompt = (
            "Extract all entities (persons, organizations, locations, dates) from the following text "
            "and provide the output in the specified JSON format with type, name, and short description of what the entity represents taken from the text for each entity."
        )
        for idx, node in enumerate(documents):
            if node.metadata.get("type") in ["section", "subsection"]:
                response = self.openai_request(prompt, node.text)

                if response:
                    entities_json = self.get_response(response)
                    print(entities_json)

                    if entities_json:
                        entity_node = TextNode(
                            text=entities_json,
                            metadata={
                                "id": f"{node.metadata['id']}_entities",
                                "type": "entities",
                                "source": node.metadata["source"],
                                "parent_id": node.metadata["id"],
                            },
                        )
                        entities_nodes.append(entity_node)

        return documents + entities_nodes


# get nodes summary and save as a child node
class SummaryTransformation(OpenAIBaseTransformation):
    def __call__(self, documents, **kwargs):
        new_nodes = []

        for idx, node in enumerate(documents):
            if node.metadata.get("type") in ["section", "subsection"]:
                context = node.metadata.get("context")
                prompt = (
                    f"Summarize the following text {node.text}, taking into account given context: {context}"
                    "as output give a string of a brief summary (6 sentences) of the text."
                )

                response = self.openai_request(prompt, node.text)
                summary = self.get_response(response)
                summary_node = TextNode(
                    text=summary,
                    metadata={
                        "id": f"{node.metadata['id']}_summary",
                        "type": "summary",
                        "source": node.metadata["source"],
                        "parent_id": node.metadata["id"],
                        "context": context,
                    },
                )
                new_nodes.append(summary_node)
        return documents + new_nodes


class KeyTakeawaysTransformation(OpenAIBaseTransformation):
    def __call__(self, documents, **kwargs):
        new_nodes = []

        for idx, node in enumerate(documents):
            if node.metadata.get("type") in ["section", "subsection"]:
                context = node.metadata.get("context")
                prompt = (
                    f"Give a list of key takeaways from this text {node.text}, taking into account given context: {context}"
                    "as output give a string of a list of key takeaways from the text."
                )

                response = self.openai_request(prompt, node.text)
                takeways = self.get_response(response)
                takeaways_node = TextNode(
                    text=takeways,
                    metadata={
                        "id": f"{node.metadata['id']}_takeaways",
                        "type": "key_takeaways",
                        "source": node.metadata["source"],
                        "parent_id": node.metadata["id"],
                        "context": context,
                    },
                )
                new_nodes.append(takeaways_node)
        return documents + new_nodes
