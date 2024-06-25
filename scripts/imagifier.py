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

load_dotenv()
USER_AGENT = os.getenv("USER_AGENT")
CONFIG_CAIRO_PATH = os.getenv("CONFIG_CAIRO_PATH")


try:
    ctypes.CDLL(CONFIG_CAIRO_PATH)
except OSError as e:
    print(f"Error loading libcairo-2.dll: {e}")
import cairosvg


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


def convert_images_to_png(page, min_size=(50, 50)):
    """
    Download all images from a wiki page and convert them to PNG format

    Parameters
        page_title : string
            title of the wiki page
        url : string
            URL of the MediaWiki API
        min_size : tuple
            minimum size of the image to be downloaded

    Returns
        None
    """

    images = page.images

    headers = {"User-Agent": USER_AGENT}

    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(script_dir, "../data/images")
    svg_dir = os.path.join(script_dir, "../data/images/svg")
    png_dir = os.path.join(script_dir, "../data/images/png")

    png_images = []

    for image_url in images:

        try:
            response = requests.get(image_url, headers=headers)
            response.raise_for_status()
            image_data = response.content
            image_name = os.path.basename(image_url)
            image_name = clean_filename(image_name)
            image_name_without_ext = os.path.splitext(image_name)[0]
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(svg_dir, exist_ok=True)
            os.makedirs(png_dir, exist_ok=True)

            if is_image_too_small(image_data, min_size):
                print(f"Skipping image {image_name} as it is too small.")
                continue

            if image_url.endswith(".svg"):
                svg_path = f"{svg_dir}/{image_name_without_ext}.svg"
                save_svg(image_data, svg_path)

                preprocessed_svg = preprocess_svg(image_data.decode("utf-8"))

                try:
                    cairosvg.svg2png(
                        bytestring=preprocessed_svg.encode("utf-8"),
                        write_to=f"{png_dir}/{image_name_without_ext}.png",
                    )
                    print(f"Converted SVG to PNG: {image_name_without_ext}.png")

                except Exception as e:
                    print(
                        f"Error converting SVG to PNG for URL: {image_url}. Error: {e}"
                    )
            else:
                try:
                    image = Image.open(io.BytesIO(image_data))
                    if image.format != "PNG":
                        image.save(f"{png_dir}/{image_name_without_ext}.png", "PNG")
                        print(f"Converted image to PNG: {image_name_without_ext}.png")
                    else:
                        save_image(
                            image_data,
                            f"{png_dir}/{image_name_without_ext}.png",
                        )
                        print(f"Saved PNG image: {image_name_without_ext}.png")
                except UnidentifiedImageError:
                    print(f"Unable to identify image at URL: {image_url}")
                    save_image(
                        image_data,
                        f"{png_dir}/{image_name_without_ext}_error.png",
                    )
                    print(
                        f"Saved image with error to check: {image_name_without_ext}_error.png"
                    )
        except requests.exceptions.RequestException as e:
            print(f"Error downloading image from URL: {image_url}. Error: {e}")
