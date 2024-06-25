from openai import AzureOpenAI
from dotenv import load_dotenv
import os
import requests
import base64
from PIL import Image
import io


load_dotenv()
GPT4o_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ENDPOINT = os.getenv("OPENAI_ENDPOINT")
GPT4o_DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_ID")
api_version = "2024-02-01"
GPT4V_ENDPOINT = f"{OPENAI_ENDPOINT}/openai/deployments/{GPT4o_DEPLOYMENT_NAME}/chat/completions?api-version=2024-02-15-preview"


headers = {
    "Content-Type": "application/json",
    "api-key": GPT4o_API_KEY,
}


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
        response = requests.post(GPT4V_ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        raise SystemExit(f"Failed to make the request. Error: {e}")


test_text = "In 1802, the British began the Great Trigonometrical Survey of India to fix the locations, heights, and names of the world's highest mountains. Starting in southern India, the survey teams moved northward using giant theodolites, each weighing 500 kg (1,100 lb) and requiring 12 men to carry, to measure heights as accurately as possible. They reached the Himalayan foothills by the 1830s, but Nepal was unwilling to allow the British to enter the country due to suspicions of their intentions. Several requests by the surveyors to enter Nepal were denied."


def get_summary(response):
    try:
        return response["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        return "Classification failed or unclear"
