from mediawiki import MediaWiki
import re
import os
import logging
from dotenv import load_dotenv


# get env variables
load_dotenv()
url = os.getenv("WIKI_API_URL")
user_agent = os.getenv("WIKI_USER_AGENT")

# initialise the mediawiki object
wikipedia = MediaWiki(user_agent=user_agent)

# set the api url if exists, otherwise use default wikipedia
if url and url.strip():
        try:
            wikipedia.set_api_url(url)
        except Exception as e:
            logging.error(f"Error setting API URL: {e}. Defaulting to Wikipedia.")
            wikipedia.set_api_url("https://en.wikipedia.org/w/api.php")
else:
        logging.warning("Empty or invalid API URL. Defaulting to Wikipedia.")
        wikipedia.set_api_url("https://en.wikipedia.org/w/api.php")


def get_wiki_page(title):
    """initialises the page

    Parameters
        title : string
            the title of the page

    Returns
        page: MediaWikiPage
            mediawikipage object
            if error bc of not specified enough title - part of the error message are the possible titles
    """

    try:
        page = wikipedia.page(title)
        return page
    except Exception as e:
        logging.error(f"Error retrieving page {title}: {e}")
        return None


def get_page_categories(page):
    """get categories of the page

    Parameters
        page : MediaWikiPage
            the mediawikipage object

    Returns:
        categories : list
            list of categories of the page
    """
    try:
        return page.categories
    except Exception as e:
        logging.error(f"Failed to retrieve page categories: {e}")
        return []


def get_page_content(page):
    """get full content

    Parameters
        page : MediaWikiPage
            the mediawikipage object

    Returns:
        content : str
            text content of the page with sections and subsections separated by === section ===

    """
    try:
        return page.content
    except Exception as e:
        logging.error(f"Failed to retrieve page content: {e}")
        return ""


def get_page_sections(page):
    """get sections of the page

    Parameters
        page : MediaWikiPage
            the mediawikipage object

    Returns:
        sections : list
            list of sections in the page
    """
    try:
        return page.sections
    except Exception as e:
        logging.error(f"Failed to retrieve page sections: {e}")
        return []


def get_section_content(page, section_title):
    """Get content of a specific section

    Parameters
        page : MediaWikiPage
            mediaWikiPage object.
        section_title : str
            title of the section

    Returns:
        content : str
            content of the section
    """
    try:
        return page.section(section_title)
    except Exception as e:
        logging.error(f"Failed to retrieve section '{section_title}' content: {e}")
        return ""


def get_page_html(page):
    """get html of the page

    Parameters
        page : MediaWikiPage
            the mediawikipage object

    Returns:
        html : str
            html content of the page
    """
    try:
        return page.html
    except Exception as e:
        logging.error(f"Failed to retrieve page HTML: {e}")
        return ""


def get_intro_content(page_content):
    """
    Extract the introductory content from a Wikipedia page (that is not in sections)

    Parameters
        page_content : str
            The full content of the Wikipedia page.

    Returns
        intro_content : str
            The introductory content of the page.
    """
    # Regex pattern to identify sections and subsections
    section_pattern = re.compile(r"(==[^=].*?==)|(===.*?===)", re.MULTILINE)

    # Split content into sections
    splits = section_pattern.split(page_content)

    # Intro content is everything before the first section
    intro_content = splits[0].strip() if splits else ""

    return intro_content


# splitting the sectins and subsections to see the hierarchical structure of the page - contents
def extract_section_titles(page_content):
    # identify pattern with == section == or === subsection ===
    section_pattern = re.compile(r"(^==[^=].*?==)|(^===.*?===)", re.MULTILINE)

    # find all
    titles = section_pattern.findall(page_content)

    sections = []
    current_section = None
    current_subsections = []

    for title in titles:
        section_title = title[0].strip("= ").strip() if title[0] else None
        subsection_title = title[1].strip("= ").strip() if title[1] else None

        if section_title:
            if current_section:
                sections.append((current_section, current_subsections))
            current_section = section_title
            current_subsections = []
        elif subsection_title:
            current_subsections.append(subsection_title)

    if current_section:
        sections.append((current_section, current_subsections))

    return sections






