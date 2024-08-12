from openai import AzureOpenAI
from dotenv import load_dotenv
import os
import requests
import base64
from PIL import Image
import io
import logging
from scripts.helper import load_env
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential_jitter,
    retry_if_exception_type,
    before_sleep_log,
)
import time


load_dotenv()

# get env variables
env_vars = load_env(
    "AZURE_OPENAI_API_KEY",
    "OPENAI_ENDPOINT",
    "GPT4O_DEPLOYMENT_ID",
    "GPT4O_API_VERSION",
)

GPT4_ENDPOINT = f'{env_vars["OPENAI_ENDPOINT"]}/openai/deployments/{env_vars["GPT4O_DEPLOYMENT_ID"]}/chat/completions?api-version={env_vars["GPT4O_API_VERSION"]}'


headers = {
    "Content-Type": "application/json",
    "api-key": env_vars["AZURE_OPENAI_API_KEY"],
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
                        "text": "Please classify the following image as either a plot (including plots, graphs, diagrams) or an actual image and output the classification class only, in one word format: plot or image",
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

    # Send request to api
    try:
        response = requests.post(GPT4_ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()
    except requests.RequestException as e:
        raise SystemExit(f"Failed to make the request. Error: {e}")

    # response
    response_json = response.json()
    print(response_json)
    return response_json


def resize_image_if_large(base64_image, max_size=(1024, 1024)):
    """
    Resize the image if it's larger than the max size while keeping it as PNG.

    :param base64_image: Original image data in base64 format.
    :param max_size: Maximum width and height of the resized image.
    :return: Resized image data in base64 format.
    """
    try:
        image_data = base64.b64decode(base64_image)

        with Image.open(io.BytesIO(image_data)) as img:
            if img.size[0] * img.size[1] > max_size[0] * max_size[1]:
                img.thumbnail(max_size, Image.LANCZOS)

                # Save the resized image to a bytes buffer
                buffer = io.BytesIO()
                img.save(buffer, format="PNG", optimize=True)

                resized_image_data = buffer.getvalue()
                return base64.b64encode(resized_image_data).decode("utf-8")

        return base64_image

    except Exception as e:
        logging.error(f"Error resizing image: {e}")
        return base64_image


@retry(
    wait=wait_exponential_jitter(initial=1, max=60),
    stop=stop_after_attempt(10),
    retry=retry_if_exception_type(requests.exceptions.RequestException),
    before_sleep=before_sleep_log(logging, logging.WARNING),
)
def classify_image_from_memory(image_data):

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
                        "text": "Please classify the following image as either a plot (including plots, graphs, diagrams) or an actual image and output the classification class only.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_data}"},
                    },
                ],
            },
        ],
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 800,
    }
    response = None
    try:
        response = requests.post(GPT4_ENDPOINT, headers=headers, json=payload)
        logging.info(f"Image classification response: {response.json()}")
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        if response is not None:
            if response.status_code == 429:
                logging.warning("Rate limit exceeded. Retrying...")
                time.sleep(10)
            elif response.status_code == 400 and "image is too large" in response.text:
                logging.warning("Image is too large. Resizing and retrying...")
                resized_image = resize_image_if_large(image_data)
                return classify_image_from_memory(resized_image)

        logging.error(f"Failed to classify the image. Error: {e}")
        raise

    return response.json()


# Extract and print classification result
def get_classification(response):
    try:
        return response["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        return "Classification failed or unclear"


def classify_and_update_image_type(image_data):
    response = classify_image_from_memory(image_data)
    logging.info(f"Image classification response: {response}")
    classification = get_classification(response)
    return classification.lower() if classification else "unknown"
