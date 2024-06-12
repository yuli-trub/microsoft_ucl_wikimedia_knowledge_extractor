from mediawiki import MediaWiki

wikipedia = MediaWiki(user_agent="KnowledgeExtractor/1.0 (ucabytr@ucl.ac.uk)")


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
        print(f"Error retrieving page {title}: {e}")
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
        print(f"Failed to retrieve page categories: {e}")
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
        print(f"Failed to retrieve page content: {e}")
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
        print(f"Failed to retrieve page sections: {e}")
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
        print(f"Failed to retrieve section '{section_title}' content: {e}")
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
        print(f"Failed to retrieve page HTML: {e}")
        return ""


# print(get_page_html(get_wiki_page("Python (programming language)")))
print(get_page_categories((get_wiki_page("Mount Everest"))))
