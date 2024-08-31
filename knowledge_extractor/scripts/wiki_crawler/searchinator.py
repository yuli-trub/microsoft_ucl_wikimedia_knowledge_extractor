from mediawiki import MediaWiki
import logging
import os
from dotenv import load_dotenv

# get env variables
load_dotenv()
user_agent = os.getenv("WIKI_USER_AGENT")

def search_wiki(query, url=None, results=2, suggestions=False):
    """
    Search for Wikipedia pages matching the query

    Parameters
        query : string
            search term to query wiki
        url : string, optional
            custom wiki API URL
        results : int, optional
            max number of results to return. Default = 10
        suggestions : bool, optional
            whether to include suggestions. Default = False

    Returns
        list
            list of top 10 relevant of Wikipedia page titles matching the query
    """
    wikipedia = MediaWiki(user_agent=user_agent)
    if url:
        try:
            wikipedia.set_api_url(url)
        except Exception as e:
            logging.error(f"Error setting API URL: {e}. Defaulting to Wikipedia.")
            wikipedia.set_api_url("https://en.wikipedia.org/w/api.php")

    try:
        results = wikipedia.search(query, results=results, suggestion=suggestions)
        return results
    except Exception as e:
        logging.error(f"Error searching Wikipedia: {e}")
        return []
