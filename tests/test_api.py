import requests
from unittest.mock import patch
from knowledge_extractor.scripts.wiki_crawler.data_fetcher import fetch_wiki_data

# Mock test for the API call
def test_fetch_wiki_data():
    mock_response = {
        "title": "Eiffel Tower",
        "extract": "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France."
    }

    # simulate a successful API call
    with patch('requests.get') as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = mock_response

        data = fetch_wiki_data("Eiffel Tower")

        # Verify that the returned data matches the mock response
        assert data["title"] == "Eiffel Tower"
        assert "Eiffel Tower is a wrought-iron lattice tower" in data["extract"]
