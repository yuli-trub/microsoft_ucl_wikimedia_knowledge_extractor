from mediawiki import MediaWiki
import requests
from PIL import Image
from PIL import UnidentifiedImageError
import os
import io
from bs4 import BeautifulSoup
import re
import base64
import sys
import ctypes
from openai import AzureOpenAI
from dotenv import load_dotenv
import logging
import gc
from helper import log_duration

load_dotenv()
USER_AGENT = os.getenv("USER_AGENT")
CONFIG_CAIRO_PATH = os.getenv("CONFIG_CAIRO_PATH")


try:
    ctypes.CDLL(CONFIG_CAIRO_PATH)
except OSError as e:
    print(f"Error loading libcairo-2.dll: {e}")
import cairosvg


Image.MAX_IMAGE_PIXELS = None


def save_svg(svg_content, file_path):
    with open(file_path, "wb") as file:
        file.write(svg_content)


def preprocess_svg(svg_content):
    try:
        soup = BeautifulSoup(svg_content, "xml")
        svg_tag = soup.find("svg")
        if "width" not in svg_tag.attrs or "height" not in svg_tag.attrs:
            svg_tag["width"] = "1000"
            svg_tag["height"] = "1000"
        return str(soup)
    except Exception as e:
        print(f"Error preprocessing SVG: {e}")
        return svg_content


def save_image(image_data, file_path):
    with open(file_path, "wb") as file:
        file.write(image_data)


def is_image_too_small(image_data, min_size=(50, 50)):
    try:
        image = Image.open(io.BytesIO(image_data))
        return image.size[0] < min_size[0] or image.size[1] < min_size[1]
    except UnidentifiedImageError:
        return False


def clean_filename(filename):
    return re.sub(r'[\\/*?:"<>|]', "", filename)


def encode_image_from_file(image_path):
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
        return base64.b64encode(image_data).decode("utf-8")


def encode_image_from_memory(image_data):
    return base64.b64encode(image_data).decode("utf-8")


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


# def convert_images_to_png(page, min_size=(50, 50), batch_size=10):
#     """
#     Download all images from a wiki page and convert them to PNG format

#     Parameters
#         page_title : string
#             title of the wiki page
#         url : string
#             URL of the MediaWiki API
#         min_size : tuple
#             minimum size of the image to be downloaded

#     Returns
#         None
#     """

#     images = page.images

#     headers = {"User-Agent": USER_AGENT}

#     png_images = []

#     for i in range(0, len(images), batch_size):
#         batch = images[i : i + batch_size]
#         for image_url in batch:
#             try:
#                 response = requests.get(image_url, headers=headers, stream=True)
#                 response.raise_for_status()
#                 image_data = response.content
#                 image_name = os.path.basename(image_url)
#                 image_name = clean_filename(image_name)
#                 image_name_without_ext = os.path.splitext(image_name)[0]

#                 if is_image_too_small(image_data, min_size):
#                     logging.info(
#                         f"Skipping image {image_name} as it is too small."
#                     )
#                     continue

#                 if image_url.endswith(".svg"):
#                     preprocessed_svg = preprocess_svg(image_data.decode("utf-8"))
#                     try:

#                         png_data = cairosvg.svg2png(
#                             bytestring=preprocessed_svg.encode("utf-8")
#                         )
#                         png_images.append(
#                             {
#                                 "image_data": base64.b64encode(png_data).decode(
#                                     "utf-8"
#                                 ),
#                                 "image_name": image_name_without_ext,
#                             }
#                         )
#                         logging.info(
#                             f"Converted SVG to PNG: {image_name_without_ext}"
#                         )
#                     except Exception as e:
#                         print(
#                             f"Error converting SVG to PNG for URL: {image_url}. Error: {e}"
#                         )
#                 else:
#                     try:
#                         image_data = resize_image_if_large(image_data)
#                         image = Image.open(io.BytesIO(image_data))
#                         png_buffer = io.BytesIO()
#                         if image.format != "PNG":
#                             if image.mode == "CMYK":
#                                 image = image.convert("RGB")
#                             image.save(png_buffer, format="PNG")
#                             png_data = png_buffer.getvalue()
#                             png_images.append(
#                                 {
#                                     "image_data": base64.b64encode(png_data).decode(
#                                         "utf-8"
#                                     ),
#                                     "image_name": image_name_without_ext,
#                                 }
#                             )
#                             logging.info(
#                                 f"Converted image to PNG: {image_name_without_ext}"
#                             )
#                         else:
#                             png_images.append(
#                                 {
#                                     "image_data": base64.b64encode(image_data).decode(
#                                         "utf-8"
#                                     ),
#                                     "image_name": image_name_without_ext,
#                                 }
#                             )
#                             logging.info(
#                                 f"Saved PNG image: {image_name_without_ext}"
#                             )
#                     except UnidentifiedImageError:
#                         print(f"Unable to identify image at URL: {image_url}")

#             except requests.exceptions.RequestException as e:
#                 print(f"Error downloading image from URL: {image_url}. Error: {e}")

#         del batch
#         gc.collect()
#     logging.info(
#         f"Total images converted: {len(png_images)}"
#     )  # todo: remove later

#     return png_images


@log_duration
def process_image(image_url, headers, min_size):
    try:
        response = requests.get(image_url, headers=headers, stream=True)
        response.raise_for_status()
        image_data = response.content
        image_name = os.path.basename(image_url)
        image_name = clean_filename(image_name)
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
                    }
                else:
                    logging.info(f"Saved PNG image: {image_name_without_ext}")
                    return {
                        "image_data": base64.b64encode(image_data).decode("utf-8"),
                        "image_name": image_name_without_ext,
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
