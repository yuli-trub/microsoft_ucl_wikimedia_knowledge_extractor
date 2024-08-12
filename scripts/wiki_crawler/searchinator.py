from mediawiki import MediaWiki


def search_wiki(query, url=None, results=4, suggestions=False):
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
    wikipedia = MediaWiki(user_agent="KnowledgeExtractor/1.0 (ucabytr@ucl.ac.uk)")
    if url:
        try:
            wikipedia.set_api_url(url)
        except Exception as e:
            print(f"Error setting API URL: {e}. Defaulting to Wikipedia.")
            wikipedia.set_api_url("https://en.wikipedia.org/w/api.php")

    try:
        results = wikipedia.search(query, results=results, suggestion=suggestions)
        return results
    except Exception as e:
        print(f"Error searching Wikipedia: {e}")
        return []


# print(search_wiki("Spider-Man", url="https://marvel.fandom.com/api.php"))
