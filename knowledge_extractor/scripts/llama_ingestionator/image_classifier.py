from dotenv import load_dotenv
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
    after_log,)
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

# set headers
headers = {
    "Content-Type": "application/json",
    "api-key": env_vars["AZURE_OPENAI_API_KEY"],
}


def resize_image_if_large(base64_image, max_size=(1024, 1024)):
    """
    Resize the image if it's larger than the max size while keeping it as PNG.

    args:  
        base64_image: str - base64 encoded image
        max_size: tuple - maximum size of the image

    Returns:
        str: base64 encoded image
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
    after=after_log(logging, logging.ERROR)
)
def classify_image(image_data, image_name):
    """ Classify an image as a plot or an actual image using the AI.
    
    Args:
        image_data (str): base64 encoded image
        image_name (str): name of the image

    Returns:
        str: classification of the image
    """

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
                        "text": f"Please classify the following image of {image_name} as either a plot (including plots, graphs, diagrams) or an actual image and output the classification class only.",
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
        response.raise_for_status()
        response_json = response.json()
        classification = response_json["choices"][0]["message"]["content"]

    except requests.exceptions.RequestException as e:
        if response is not None:
            response_json = response.json()
            if response.status_code == 429:
                logging.warning("Rate limit exceeded. Retrying...")
                time.sleep(10)
            elif response.status_code == 400 and "image is too large" in response.text:
                logging.warning("Image is too large. Resizing and retrying...")
                resized_image = resize_image_if_large(image_data)
                return classify_image(resized_image, image_name)
            elif response.status_code == 400  and response_json["error"].get("code", "") == "content_filter":
                logging.warning(
                    "Image classification was blocked by Azure's content management policy due to potentially sensitive content. Skipping this image."
                )
                return "sensitive_image" 

        logging.error(f"Failed to classify the image. Error: {e}")
        raise

    return classification


def classify_and_update_image_type(image_data, image_name):
    """Classify the image and return the classification."""
    classification = classify_image(image_data, image_name)
    logging.info(f"Image classification: {classification}")
    return classification.lower() if classification else "unknown"
