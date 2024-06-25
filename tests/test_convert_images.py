import unittest
from unittest.mock import patch, Mock
from mediawiki import MediaWiki
import requests
from PIL import Image
from PIL import UnidentifiedImageError
import io

from scripts.imagifier import convert_images_to_png


class TestConvertImagesToPng(unittest.TestCase):
    @patch("requests.get")
    @patch("mediawiki.MediaWiki")
    def test_convert_images_to_png(self, MockMediaWiki, mock_requests_get):
        # Mock MediaWiki page object
        mock_page = Mock()
        mock_page.images = [
            "https://example.com/image1.jpg",
            "https://example.com/image2.svg",
            "https://example.com/image3.png",
        ]

        # Mock responses for images
        mock_responses = {
            "https://example.com/image1.jpg": requests.Response(),
            "https://example.com/image2.svg": requests.Response(),
            "https://example.com/image3.png": requests.Response(),
        }

        mock_responses["https://example.com/image1.jpg"]._content = (
            self._create_sample_image_bytes(format="JPEG")
        )
        mock_responses["https://example.com/image1.jpg"].status_code = 200
        mock_responses["https://example.com/image2.svg"]._content = (
            b'<svg height="100" width="100"><circle cx="50" cy="50" r="40" stroke="black" stroke-width="3" fill="red" /></svg>'
        )
        mock_responses["https://example.com/image2.svg"].status_code = 200
        mock_responses["https://example.com/image3.png"]._content = (
            self._create_sample_image_bytes(format="PNG")
        )
        mock_responses["https://example.com/image3.png"].status_code = 200

        mock_requests_get.side_effect = lambda url, headers: mock_responses[url]

        png_images = convert_images_to_png(mock_page)

        self.assertEqual(len(png_images), 3)  # Check if three images are processed
        for img_data in png_images:
            self.assertTrue(self._is_png(img_data["image_data"]))
            self.assertTrue(isinstance(img_data["image_name"], str))

    def _create_sample_image_bytes(self, format="PNG"):
        img = Image.new("RGB", (100, 100), color=(73, 109, 137))
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format=format)
        img_byte_arr = img_byte_arr.getvalue()
        return img_byte_arr

    def _is_png(self, image_data):
        try:
            img = Image.open(io.BytesIO(image_data))
            return img.format == "PNG"
        except UnidentifiedImageError:
            return False


if __name__ == "__main__":
    unittest.main()
