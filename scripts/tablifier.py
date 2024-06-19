from mediawiki import MediaWiki
from bs4 import BeautifulSoup
import pandas as pd


USER_AGENT = "KnowledgeExtractor/1.0 (ucabytr@ucl.ac.uk) Python-requests/2.25.1"


# this could be slow if the page is large?
def get_html_page(page):
    """
    Get the html of a Wiki page

    Parameters
        page : MediaWikiPage
            the mediawikipage object

    Returns
        string
            the html content of the page
    """
    return page.html


def extract_tables(html_page, output_dir=None):
    """
    Extract tables from a Wiki page html

    Parameters
        html_page : string
            the html content of the page

    Returns
        list
            list of tables from the page
    """

    soup = BeautifulSoup(html_page, "lxml")
    tables = []

    for table in soup.find_all("table", class_="wikitable"):
        rows = table.find_all("tr")
        table_data = []

        for row in rows:
            cols = row.find_all(["td", "th"])
            cols = [col.text.strip() for col in cols]
            table_data.append(cols)

        # convert to dataframe
        df = pd.DataFrame(table_data)
        df = df.dropna(how="all", axis=1)
        df = df.dropna(how="all")
        tables.append(df)

    return tables


# test
wikipedia = MediaWiki(user_agent=USER_AGENT)
# wikipedia.set_api_url("https://marvel.fandom.com/api.php")
test_page = wikipedia.page("Mount Everest")
# print(extract_tables(get_html_page(page=test_page)))
