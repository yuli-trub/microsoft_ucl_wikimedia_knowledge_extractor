import os
import requests
from llama_index.core.schema import TransformComponent
import re
from llama_index.core.schema import TextNode, ImageNode
from llama_index.core.node_parser import SemanticSplitterNodeParser
import logging
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo
from llama_index.core import Settings
from pydantic import Field
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential_jitter,
    retry_if_exception_type,
    retry_if_exception
)
import time
from scripts.helper import load_env, log_duration
import base64
from PIL import Image
import io


# get env variables
env_vars = load_env(
    "AZURE_OPENAI_API_KEY",
    "OPENAI_ENDPOINT",
    "GPT4O_DEPLOYMENT_ID",
    "GPT4O_API_VERSION",
    "GPT4_ENDPOINT",
    "EMBEDDING_DEPLOYMENT_ID",
    "EMBEDDING_API_VERSION",
    "COMPUTER_VISION_ENDPOINT",
    "COMPUTER_VISION_API_KEY",
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
    buffer_size=1,
    breakpoint_percentile_threshold=70,
    max_tokens=7000,
    embed_model=Settings.embed_model,
)


class EmbeddingTransformation(TransformComponent):
    """ Embedding transformation component to generate embeddings for text and image nodes """
    def __call__(self, documents, text_embed_model, **kwargs):
        logging.info(f"embed model: {text_embed_model}")
        logging.info(f"Processing {len(documents)} nodes")
        text_embed_model = text_embed_model
        for doc in documents:
            if doc.metadata.get("needs_embedding"):
                logging.info(f"Generating embedding for node ID: {doc.metadata['title']}")
                if isinstance(doc, ImageNode):
                    embedding = None
                    # uncomment if want to generate image embeddings using Computer Vision API for future work
                    # embedding = self.get_image_embedding(doc.metadata["url"])
                    doc.embedding = embedding
                elif isinstance(doc, TextNode):
                    embedding = text_embed_model.get_text_embedding(doc.text)
                    doc.embedding = embedding
                    
        return documents

    @retry(
        wait=wait_exponential_jitter(initial=1, max=60),
        stop=stop_after_attempt(10),
        retry=retry_if_exception_type(requests.exceptions.RequestException),
    )
    def get_image_embedding(self, image_url):
        """ Get image embedding using Azure Computer Vision API """

        endpoint = f"{env_vars['COMPUTER_VISION_ENDPOINT']}computervision/"
        headers = {
            "Content-Type": "application/json",
            "Ocp-Apim-Subscription-Key": env_vars["COMPUTER_VISION_API_KEY"],
        }

        data = {"url": image_url}
        version = "?api-version=2024-02-01&model-version=2023-04-15"
        vectorize_img_url = endpoint + "retrieval:vectorizeImage" + version


        try:
            response = requests.post(vectorize_img_url, headers=headers, json=data)
            response.raise_for_status()
            json_data = response.json()
            return json_data.get("vector")
        except requests.exceptions.RequestException as e:
            if response:
                if response.status_code == 429:
                    logging.warning("Rate limit exceeded. Retrying...")
                    time.sleep(10)
                else:
                    logging.error(
                        f"Request failed with error: {e}. Response content: {response.content}"
                    )
            raise


class TextCleaner(TransformComponent):
    """ Text cleaner transformation component to remove special characters from text nodes """  
    def __call__(self, nodes, **kwargs):
        logging.info(f"Processing {len(nodes)} nodes for text cleaning")
        for node in nodes:
            if isinstance(node, TextNode) and node.metadata.get("type") not in [
                "page",
                "table",
                "citation",
                "archive-citation",
                "wiki-ref",
                "image",
                "plot",
            ]:
                node.text = re.sub(r"[^0-9A-Za-z ]", "", node.text)
        logging.info(f"Text cleaning completed")
        return nodes


class OpenAIBaseTransformation(TransformComponent):
    """ Base class for OpenAI transformations """
    @log_duration
    @retry(
        wait=wait_exponential_jitter(initial=1, max=60),
        stop=stop_after_attempt(10),
        retry=retry_if_exception(lambda e: not (isinstance(e, requests.exceptions.RequestException) and e.response is not None and e.response.status_code == 400 and e.response.json().get("error", {}).get("code", "") == "content_filter")),
    )
    def openai_request(self, prompt, image=None, text=None, function=None):
        """ Make a request to OpenAI API """

        user_content = [{"type": "text", "text": prompt}]

        if text:
            user_content.append({"type": "text", "text": text})

        if image:
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image}"},
                }
            )

        payload = {
            "messages": [
                {"role": "system", "content": "You are an AI assistant."},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.7,
            "top_p": 0.95,
            "max_tokens": 800,
        }
        if function:
            payload["functions"] = [function]
            payload["function_call"] = {"name": function["name"]}

        response = None
        try:
            response = requests.post(GPT4_ENDPOINT, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            response_json = response.json() 
            if response and response.status_code == 429:
                logging.warning("Rate limit exceeded. Retrying...")
                time.sleep(10)
            
            if response:
                logging.error(
                    f"Request failed with error: {e}. Response content: {response.content}"
                )
            if response.status_code == 400 and response_json.get("error", {}).get("code", "") == "content_filter":
                logging.error(
                    "The content was blocked by Azure's content management policy due to potentially sensitive content. Skipping this image."
                )
                return None
        
            raise

    def get_response(self, response):
        """ Get response from OpenAI API with logging information of tokens used and estimated cost """
        try:
            token_usage = response["usage"]["total_tokens"]
            logging.info(f"Tokens used: {token_usage}")
            cost = self.calculate_cost(token_usage)
            logging.info(f"Estimated cost: ${cost:.2f}")
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            logging.error(f"Failed to retrieve response.")
            return "Transformation failed or unclear"

    def calculate_cost(self, tokens):
        return tokens * 0.00002


class SemanticChunkingTransformation(TransformComponent):
    """ Semantic chunking transformation component to split text nodes into smaller chunks """
    def __call__(self, documents, **kwargs):
        logging.info(f"Chunking: {len(documents)} documents")

        transformed_nodes = []
        splitter = Settings.text_splitter

        for node in documents:
            if node.metadata.get("type") in ["section", "subsection"]:
                chunks = splitter.get_nodes_from_documents([node])
                logging.info(
                    f"Generated {len(chunks)} chunks for node ID: {node.node_id}"
                )
                transformed_nodes.append(node)

                # set metadata for each chunk
                for idx, chunk in enumerate(chunks):
                    chunk.metadata["title"] = f"{node.metadata['title']}_chunk_{idx}"
                    chunk.relationships[NodeRelationship.PARENT] = RelatedNodeInfo(
                        node_id=node.node_id
                    )
                    chunk.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(
                    node_id=node.relationships[NodeRelationship.SOURCE].node_id
                )
                    chunk.metadata["type"] = "chunk"
                    chunk.metadata["needs_embedding"] = True
                    transformed_nodes.append(chunk)
            else:

                # set the flag for embedding
                if node.metadata.get("type") not in [
                    "entities",
                    "summary",
                    "key_takeaways",
                    "image_description",
                    "plot_insights",
                    "image_entities",
                    "chunk",
                    "image",
                    "plot",
                ]:
                    node.metadata["needs_embedding"] = False
                elif node.metadata.get("type") in [
                    "image",
                    "plot",
                ]:
                    node.metadata["needs_embedding"] = True
                transformed_nodes.append(node)

        # to avoid terminating the pipeline if there are no transfromed nodes
        for node in transformed_nodes:
            node.metadata["last_transformed"] = str(time.time())

        return transformed_nodes


class EntityExtractorTransformation(OpenAIBaseTransformation):
    """ Entity extraction transformation component to extract entities from text nodes """
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
            logging.info(f"Extracting entities from text node ID: {node.metadata['title']}")
            if node.metadata.get("type") in ["section", "subsection"]:
                response = self.openai_request(prompt, text=node.text)

                if response:
                    entities_json = self.get_response(response)

                    if entities_json:
                        entity_node = TextNode(
                            text=entities_json,
                            metadata={
                                "title": f"{node.metadata['title']}_entities",
                                "type": "entities",
                                "needs_embedding": True,
                            },
                        )
                        entity_node.relationships[NodeRelationship.PARENT] = (
                            RelatedNodeInfo(node_id=node.node_id)
                        )
                        source_id = node.relationships[NodeRelationship.SOURCE].node_id
                        entity_node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(
                            node_id=source_id
                        )   
                        entities_nodes.append(entity_node)
        logging.info(f"Extracted entities")

        transformed_nodes = documents + entities_nodes


        # to avoid terminating the pipeline if there are no transfromed nodes
        for node in transformed_nodes:
            node.metadata["last_transformed"] = str(time.time())

        return transformed_nodes


class SummaryTransformation(OpenAIBaseTransformation):
    """ Summary transformation component to generate summaries for text nodes """
    def __call__(self, documents, **kwargs):
        new_nodes = []

        for idx, node in enumerate(documents):
            if node.metadata.get("type") in ["section", "subsection"]:
                logging.info(f"Summarising node ID: {node.node_id}")
                context = node.metadata.get("context")
                prompt = (
                    f"Summarise the following text {node.text}, taking into account given context: {context}"
                    "as output give a string of a brief summary (6 sentences) of the text."
                )

                response = self.openai_request(prompt, text=node.text)
                summary = self.get_response(response)
                summary_node = TextNode(
                    text=summary,
                    metadata={
                        "title": f"{node.metadata['title']}_summary",
                        "type": "summary",
                        "needs_embedding": True,
                        "context": context,
                    },
                )
                summary_node.relationships[NodeRelationship.PARENT] = RelatedNodeInfo(
                    node_id=node.node_id
                )
                source_id = node.relationships[NodeRelationship.SOURCE].node_id
                summary_node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(
                    node_id=source_id
                )
                new_nodes.append(summary_node)

        transformed_nodes = documents + new_nodes

        # to avoid terminating the pipeline if there are no transfromed nodes
        for node in transformed_nodes:
            node.metadata["last_transformed"] = str(time.time())

        return transformed_nodes


class KeyTakeawaysTransformation(OpenAIBaseTransformation):
    """ Key takeaways transformation component to extract key takeaways from text nodes """
    def __call__(self, documents, **kwargs):
        new_nodes = []

        for idx, node in enumerate(documents):
            if node.metadata.get("type") in ["section", "subsection"]:
                logging.info(f"Extracting key takeaways from text node ID: {node.metadata['title']}")
                context = node.metadata.get("context")
                prompt = (
                    f"Give a list of key takeaways from this text {node.text}, taking into account given context: {context}"
                    "as output give a string of a list of key takeaways from the text."
                )
                response = self.openai_request(prompt, text=node.text)
                if response:
                    takeways = self.get_response(response)
                    takeaways_node = TextNode(
                        text=takeways,
                        metadata={
                            "title": f"{node.metadata['title']}_takeaways",
                            "type": "key_takeaways",
                            "needs_embedding": True,
                            "context": context,
                        },
                    )
                    takeaways_node.relationships[NodeRelationship.PARENT] = RelatedNodeInfo(
                        node_id=node.node_id
                    )
                    source_id = node.relationships[NodeRelationship.SOURCE].node_id
                    takeaways_node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(
                    node_id=source_id
                )
                    new_nodes.append(takeaways_node)

        transformed_nodes = documents + new_nodes

        # to avoid terminating the pipeline if there are no transfromed nodes
        for node in transformed_nodes:
            node.metadata["last_transformed"] = str(time.time())

        return transformed_nodes


def resize_image(image_base64, max_size=(1024, 1024)):
    """ Resize image to max size """
    image_data = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image_data))
    image.thumbnail(max_size, Image.LANCZOS)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


class ImageDescriptionTransformation(OpenAIBaseTransformation):
    ''' Image description transformation component to generate descriptions for image nodes '''
    def __call__(self, documents, **kwargs):
        new_nodes = []
        total_images = sum(1 for node in documents if isinstance(node, ImageNode) and node.metadata.get("type") == "image")
        total_plots = sum(1 for node in documents if isinstance(node, ImageNode) and node.metadata.get("type") == "plot")
        processed_images = 0
        processed_plots = 0

        for idx, node in enumerate(documents):
            logging.info(f"Processing documents for image description. Total documents: {len(documents)}")

            if isinstance(node, ImageNode):
                logging.info(
                    f"Creating description for image: {node.metadata['title']}"
                )
                context = node.metadata.get("context")
            
                if node.metadata.get("type") == "image":
                    prompt = f"""Considering the context for the image: {context}.
                                 Please describe the image in detail, covering the main elements visible in the picture. 
                                 Mention the setting, any people or objects of interest, and their interactions or relationships. 
                                 Highlight any emotions or atmospheres conveyed by the image, and speculate on the context or story behind what is depicted. 
                                 If certain aspects are unclear, provide a brief description of the elements that are clear and meaningful. 
                                 If you cannot provide a complete answer, specify which aspects are unclear or missing, and then state "error: unable to provide an answer
                                 Avoid stating "error: unable to provide answer" unless absolutely no analysis can be provided."""
                    processed_images += 1

                elif node.metadata.get("type") == "plot":
                    prompt = f"""Considering the context for the image: {context}. 
                                 Provide a detailed analysis of the image, identifying and describing any data, labels, or key elements visible.
                                 Explain the relationships, trends, or patterns depicted.
                                 If some parts are unclear, summarize the insights that are clear and significant.
                                 If you cannot provide a complete answer, specify which aspects are unclear or missing, and then state "error: unable to provide an answer
                                 Avoid stating "error: unable to provide answer" unless absolutely no analysis can be provided.    """
                    processed_plots += 1

                resised_image = resize_image(node.image)
                response = self.openai_request(prompt, image=resised_image)
                if response:
                    description = self.get_response(response)

                    if "error: unable" not in description:
                        description_node = TextNode(
                            text=description,
                            metadata={
                                "title": f"{node.metadata['title']}_image_description",
                                "type": "image_description",
                                "needs_embedding": True,
                                "context": context,
                            },
                        )
                        description_node.relationships[NodeRelationship.PARENT] = (
                            RelatedNodeInfo(node_id=node.node_id)
                        )
                        source_id = node.relationships[NodeRelationship.SOURCE].node_id
                        description_node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(
                         node_id=source_id
                        )
                        if node.metadata.get("type") == "image":
                            logging.info(f"Processed {processed_images} out of {total_images} images")
                        elif node.metadata.get("type") == "plot":
                            logging.info(f"Processed {processed_plots} out of {total_plots} plots")

                        new_nodes.append(description_node)

        transformed_nodes = documents + new_nodes

        # to avoid terminating the pipeline if there are no transfromed nodes
        for node in transformed_nodes:
            node.metadata["last_transformed"] = str(time.time())

        return transformed_nodes

class PlotInsightsTransformation(OpenAIBaseTransformation):
    ''' Plot insights transformation component to generate insights for plot nodes '''
    def __call__(self, documents, **kwargs):
        logging.info(f"Processing documents for plot insights. Total documents: {len(documents)}")
        new_nodes = []
        plots_found = False

        for idx, node in enumerate(documents):
            if isinstance(node, ImageNode) and node.metadata.get("type") == "plot":
                plots_found = True
                logging.info(f"Extracting insights for plot: {node.metadata['title']}")
                context = node.metadata.get("context")

                prompt = f"""
                Considering the context for the image: {context}. 
                Please analyse the diagram and summarise the key insights. 
                Identify the most significant data points and trends shown in the image. 
                If any information is unclear or missing, provide a brief explanation of the insights that are clear.
                In case you cannot provide a comprehensive answer, do not make things up. 
                If you cannot provide a complete answer, specify which aspects are unclear or missing, and then state "error: unable to provide an answer
                Avoid stating "error: unable to provide answer" unless absolutely no analysis can be provided."""

                resised_image = resize_image(node.image)
                response = self.openai_request(prompt, image=resised_image)
                if response:
                    insights = self.get_response(response)

                    if "error: unable" not in insights:
                        insights_node = TextNode(
                            text=insights,
                            metadata={
                                "title": f"{node.metadata['title']}_plot_insights",
                                "type": "plot_insights",
                                "needs_embedding": True,
                                "context": context,
                            },
                        )
                        insights_node.relationships[NodeRelationship.PARENT] = (
                            RelatedNodeInfo(node_id=node.node_id)
                        )
                        source_id = node.relationships[NodeRelationship.SOURCE].node_id
                        insights_node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(
                         node_id=source_id
                        )
                        logging.info(f"Processed {idx+1} out of {len(documents)} plots")
                        new_nodes.append(insights_node)
            else:
                continue

        if not plots_found:
            logging.info("No plots found in the documents. Skipping plot insights extraction.")
            output_docs = documents + new_nodes

            for node in output_docs:
                node.metadata["last_transformed"] = str(time.time())

            return output_docs
        
        else:
            output_docs = documents + new_nodes

            for node in output_docs:
                node.metadata["last_transformed"] = str(time.time())

            return output_docs


class ImageEntitiesTransformation(OpenAIBaseTransformation):
    ''' Image entities transformation component to extract entities from image nodes '''
    def __call__(self, documents, **kwargs):
        new_nodes = []

        for idx, node in enumerate(documents):
            if isinstance(node, ImageNode) and node.metadata.get("type") == "image":
                logging.info(f"Extracting insights for image: {node.metadata['title']}")
                context = node.metadata.get("context")

                prompt = f"""Given the image, analyze and list all important entities present.
                    Describe each entity in detail, focusing on their characteristics and significance within the provided context: {context}.
                    If certain entities or details are unclear, provide as much accurate information as you can about the elements that are clear and meaningful.
                    Avoid stating "error: unable to provide answer" unless absolutely no analysis can be provided. 
                    Ensure your response is clear and concise, highlighting any insights or observations, even if they are partial."""

                resised_image = resize_image(node.image)
                response = self.openai_request(prompt, image=resised_image)
                if response:
                    entities = self.get_response(response)

                    if "error: unable" not in entities:
                        entities_node = TextNode(
                            text=entities,
                            metadata={
                                "title": f"{node.metadata['title']}_image_entities",
                                "type": "image_entities",
                                "needs_embedding": True,
                                "context": context,
                            },
                        )
                        entities_node.relationships[NodeRelationship.PARENT] = (
                            RelatedNodeInfo(node_id=node.node_id)
                        )
                        source_id = node.relationships[NodeRelationship.SOURCE].node_id
                        entities_node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(
                         node_id=source_id
                        )
                        logging.info(f"Processed {idx+1} out of {len(documents)} images")

                        new_nodes.append(entities_node)

        transformed_nodes = documents + new_nodes

        # to avoid terminating the pipeline if there are no transfromed nodes
        for node in transformed_nodes:
            node.metadata["last_transformed"] = str(time.time())

        return transformed_nodes


class TableAnalysisTransformation(OpenAIBaseTransformation):
    ''' Table analysis transformation component to generate insights for table nodes '''
    def __call__(self, documents, **kwargs):
        new_nodes = []

        for idx, node in enumerate(documents):
            if node.metadata.get("type") == "table":
                logging.info(f"Analysinng table from node: {node.metadata['title']}")
                context = node.metadata.get("context")
                prompt = (
                    f"Analyse the following table data and provide a list of key insights and observations, considering the context: {context}. "
                     "Please focus on identifying significant trends, patterns, or anomalies, and include any important numerical values that support your analysis. "
                     "Your output should be a concise and well-structured list of key points, each highlighting an important aspect of the table data. "
                     "If applicable, include comparisons between different data points or categories and mention any outliers or unexpected results."
)

                response = self.openai_request(prompt, text=node.text)
                if response:
                    takeways = self.get_response(response)
                    takeaways_node = TextNode(
                        text=takeways,
                        metadata={
                            "title": f"{node.metadata['title']}_analysis",
                            "type": "table_analysis",
                            "needs_embedding": True,
                            "context": context,
                        },
                    )
                    takeaways_node.relationships[NodeRelationship.PARENT] = RelatedNodeInfo(
                        node_id=node.node_id
                    )
                    source_id = node.relationships[NodeRelationship.SOURCE].node_id
                    takeaways_node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(
                         node_id=source_id
                        )
                    logging.info(f"Processed {idx+1} out of {len(documents)} tables")
                    new_nodes.append(takeaways_node)

        transformed_nodes = documents + new_nodes

        # to avoid terminating the pipeline if there are no transfromed nodes
        for node in transformed_nodes:
            node.metadata["last_transformed"] = str(time.time())

        return transformed_nodes
