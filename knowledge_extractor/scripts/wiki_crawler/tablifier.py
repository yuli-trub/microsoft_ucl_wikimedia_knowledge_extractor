from mediawiki import MediaWiki
from bs4 import BeautifulSoup
import pandas as pd


def extract_tables(html_page):
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


