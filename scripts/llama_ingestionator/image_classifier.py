from openai import AzureOpenAI
from dotenv import load_dotenv
import os
import requests
import base64
from PIL import Image
import io


load_dotenv()
AZURE_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ENDPOINT = os.getenv("OPENAI_ENDPOINT")
GPT4o_DEPLOYMENT_NAME = os.getenv("GPT4O_DEPLOYMENT_ID")
api_version = "2024-02-01"
GPT4V_ENDPOINT = f"{OPENAI_ENDPOINT}/openai/deployments/{GPT4o_DEPLOYMENT_NAME}/chat/completions?api-version=2024-02-15-preview"


headers = {
    "Content-Type": "application/json",
    "api-key": AZURE_OPENAI_API_KEY,
}

# Payload test for the request from the playground
payload_test = {
    "messages": [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are an AI assistant that helps people find information.",
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "how would you define a plot?"}],
        },
    ],
    "temperature": 0.7,
    "top_p": 0.95,
    "max_tokens": 800,
}
# Send request
# try:
#     response = requests.post(GPT4V_ENDPOINT, headers=headers, json=payload_test)
#     response.raise_for_status()
# except requests.RequestException as e:
#     raise SystemExit(f"Failed to make the request. Error: {e}")


# classify image test
def encode_image_from_file(image_path):
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
        return base64.b64encode(image_data).decode("utf-8")


def classify_image_from_file(image_path):

    base64_image = encode_image_from_file(image_path)

    #  payload
    payload = {
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are an AI assistant that helps people classify images.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Please classify the following image as either a plot (including plots, graphs, diagrams) or an actual image and output the classification class only",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    },
                ],
            },
        ],
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 800,
    }

    GPT4V_ENDPOINT = f"{OPENAI_ENDPOINT}/openai/deployments/{GPT4o_DEPLOYMENT_NAME}/chat/completions?api-version=2024-02-15-preview"

    # Send request to api
    try:
        response = requests.post(GPT4V_ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()
    except requests.RequestException as e:
        raise SystemExit(f"Failed to make the request. Error: {e}")

    # response
    response_json = response.json()
    print(response_json)
    return response_json


image_path = "../data/image.png"
response = classify_image_from_file(image_path)


# Extract and print classification result
def get_classification(response):
    try:
        return response["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        return "Classification failed or unclear"


classification = get_classification(response)
print(f"Classification result: {classification}")
