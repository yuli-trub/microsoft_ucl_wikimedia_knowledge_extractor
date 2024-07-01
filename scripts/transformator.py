from openai import AzureOpenAI
from dotenv import load_dotenv
import os
import requests
import base64
from PIL import Image
import io
from llama_index.core.schema import TransformComponent
import re
from llama_index.core.schema import TextNode


load_dotenv()
GPT4o_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ENDPOINT = os.getenv("OPENAI_ENDPOINT")
GPT4o_DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_ID")
api_version = "2024-02-01"
GPT4_ENDPOINT = f"{OPENAI_ENDPOINT}/openai/deployments/{GPT4o_DEPLOYMENT_NAME}/chat/completions?api-version=2024-02-15-preview"


headers = {
    "Content-Type": "application/json",
    "api-key": GPT4o_API_KEY,
}


class OpenAIBaseTransformation(TransformComponent):
    def openai_request(self, prompt, text):
        payload = {
            "messages": [
                {"role": "system", "content": "You are an AI assistant."},
                {"role": "user", "content": f"{prompt} {text}"},
            ],
            "temperature": 0.7,
            "top_p": 0.95,
            "max_tokens": 800,
        }
        try:
            response = requests.post(GPT4_ENDPOINT, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise SystemExit(f"Failed to make the request. Error: {e}")

    def get_response(self, response):
        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            return "Transformation failed or unclear"


def general_summarisor(prompt, text):
    payload = {
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are an AI assistant that summarises the given text.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{prompt} {text}",
                    }
                ],
            },
        ],
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 800,
    }

    try:
        response = requests.post(GPT4_ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        raise SystemExit(f"Failed to make the request. Error: {e}")


test_text = "In 1802, the British began the Great Trigonometrical Survey of India to fix the locations, heights, and names of the world's highest mountains. Starting in southern India, the survey teams moved northward using giant theodolites, each weighing 500 kg (1,100 lb) and requiring 12 men to carry, to measure heights as accurately as possible. They reached the Himalayan foothills by the 1830s, but Nepal was unwilling to allow the British to enter the country due to suspicions of their intentions. Several requests by the surveyors to enter Nepal were denied."


# from llamaindex stuff - maybe remove if redundant
class TextCleaner(TransformComponent):
    def __call__(self, nodes, **kwargs):
        for node in nodes:
            node.text = re.sub(r"[^0-9A-Za-z ]", "", node.text)
        return nodes


# get nodes summary and save as a child node
class SummaryGenerator(OpenAIBaseTransformation):
    def __call__(self, nodes, **kwargs):
        for node in nodes:
            if node.metadata.get("type") == "text":
                context = node.metadata.get("context")
                prompt = f"Summarize the following text, taking into account given context: {context}"
                response = self.openai_request(prompt, node.text)
                summary = self.get_summary(response)
                summary_node = TextNode(
                    text=summary,
                    metadata={
                        "type": "summary",
                        "source": node.metadata["source"],
                        "parent_id": node.metadata["id"],
                        "context": context,
                    },
                )
                node.add_child(summary_node)
        return nodes
