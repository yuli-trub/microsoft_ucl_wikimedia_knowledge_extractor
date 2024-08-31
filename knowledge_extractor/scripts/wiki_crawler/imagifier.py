import requests
from PIL import Image
from PIL import UnidentifiedImageError
import os
import io
from bs4 import BeautifulSoup
import re
import base64
import ctypes
from dotenv import load_dotenv
import logging
import gc
from scripts.helper import log_duration
import cairosvg
import logging
from scripts.helper import sanitise_filename

load_dotenv()
USER_AGENT = os.getenv("USER_AGENT")


try:
    # Attempt to load the Cairo library for Linux
    # The Cairo library should be installed in the Docker container using a package manager
    ctypes.CDLL("libcairo.so.2")
except OSError as e:
    print(f"Error loading libcairo.so.2: {e}")


Image.MAX_IMAGE_PIXELS = None


def preprocess_svg(svg_content):
    """Preprocess SVG content by adding width and height attributes if missing."""
    try:
        soup = BeautifulSoup(svg_content, "xml")
        svg_tag = soup.find("svg")
        if "width" not in svg_tag.attrs or "height" not in svg_tag.attrs:
            svg_tag["width"] = "1000"
            svg_tag["height"] = "1000"
        return str(soup)
    except Exception as e:
        logging.error(f"Error preprocessing SVG: {e}")
        return svg_content


def is_image_too_small(image_data, min_size=(50, 50)):
    """Check if the image is too small."""
    try:
        image = Image.open(io.BytesIO(image_data))
        return image.size[0] < min_size[0] or image.size[1] < min_size[1]
    except UnidentifiedImageError:
        return False


def resize_image_if_large(image_data, max_pixels=178956970):
    """Resize the image if it exceeds the max_pixels limit."""
    try:
        image = Image.open(io.BytesIO(image_data))
        if image.size[0] * image.size[1] > max_pixels:
            ratio = (max_pixels / float(image.size[0] * image.size[1])) ** 0.5
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image.thumbnail(new_size, Image.Resampling.LANCZOS)
            output = io.BytesIO()
            image.save(output, format="PNG")
            return output.getvalue()
    except Exception as e:
        logging.error(f"Error resizing image: {e}")
    return image_data


@log_duration
def process_image(image_url, headers, min_size):
    """Download and process the image to convert it to PNG format."""
    try:
        response = requests.get(image_url, headers=headers, stream=True)
        response.raise_for_status()
        image_data = response.content
        image_name = os.path.basename(image_url)
        image_name = sanitise_filename(image_name)
        image_name_without_ext = os.path.splitext(image_name)[0]

        if is_image_too_small(image_data, min_size):
            logging.info(f"Skipping image {image_name} as it is too small.")
            return None

        if image_url.endswith(".svg"):
            preprocessed_svg = preprocess_svg(image_data.decode("utf-8"))
            try:
                png_data = cairosvg.svg2png(bytestring=preprocessed_svg.encode("utf-8"))
                logging.info(f"Converted SVG to PNG: {image_name_without_ext}")
                return {
                    # "raw_image_data": png_data,
                    "image_data": base64.b64encode(png_data).decode("utf-8"),
                    "image_name": image_name_without_ext,
                    "image_url": image_url,
                }
            except Exception as e:
                logging.error(
                    f"Error converting SVG to PNG for URL: {image_url}. Error: {e}"
                )
        else:
            try:
                image_data = resize_image_if_large(image_data)
                image = Image.open(io.BytesIO(image_data))
                png_buffer = io.BytesIO()
                if image.format != "PNG":
                    if image.mode == "CMYK":
                        image = image.convert("RGB")
                    image.save(png_buffer, format="PNG")
                    png_data = png_buffer.getvalue()
                    logging.info(f"Converted image to PNG: {image_name_without_ext}")
                    return {
                        "image_data": base64.b64encode(png_data).decode("utf-8"),
                        "image_name": image_name_without_ext,
                        "image_url": image_url,
                    }
                else:
                    logging.info(f"Saved PNG image: {image_name_without_ext}")
                    return {
                        "image_data": base64.b64encode(image_data).decode("utf-8"),
                        "image_name": image_name_without_ext,
                        "image_url": image_url,
                    }
            except UnidentifiedImageError:
                logging.error(f"Unable to identify image at URL: {image_url}")

    except requests.exceptions.RequestException as e:
        logging.error(f"Error downloading image from URL: {image_url}. Error: {e}")

    return None


def convert_images_to_png(page, min_size=(50, 50)):
    """
    Download all images from a wiki page and convert them to PNG format.

    Parameters:
        page : MediaWikiPage
            The MediaWiki page object.
        min_size : tuple
            Minimum size of the image to be downloaded.

    Returns:
        list
        List of dictionaries containing image data and image name.
    """
    images = page.images
    headers = {"User-Agent": USER_AGENT}
    png_images = []

    for image_url in images:
        png_image = process_image(image_url, headers, min_size)
        if png_image:
            png_images.append(png_image)
        # Clear memory after each image
        gc.collect()

    logging.info(f"Total images converted: {len(png_images)}")
    return png_images
