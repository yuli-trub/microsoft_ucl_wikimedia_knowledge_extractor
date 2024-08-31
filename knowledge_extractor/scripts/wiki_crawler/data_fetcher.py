from scripts.wiki_crawler.navigifier import (
    get_wiki_page,
    get_intro_content,
    extract_section_titles,
    get_page_content,
    get_page_categories,
    get_page_html,
)
from scripts.wiki_crawler.tablifier import extract_tables
from scripts.wiki_crawler.referenciator import (
    get_external_links_by_section,
    get_all_citations,
)
from scripts.wiki_crawler.imagifier import convert_images_to_png


def fetch_wiki_data(page_title):
    """Fetch all the data from a wiki page and preprocess it.
    
    Args:
        page_title (str): the title of the wiki page

    Returns:
        tuple: a tuple containing the following data:
            - page: the MediaWikiPage object
            - page_content: the content of the page
            - intro_content: the introduction content of the page
            - sections: the sections of the page
            - categories: the categories of the page
            - images: the images of the page
            - tables: the tables of the page
            - reference_dict: the references of the page
            - wiki_links_dict: the external links of the page
            - table_of_contents: the table of contents of the page
    """
    page = get_wiki_page(page_title)
    if not page:
        print(f"Failed to retrieve the page: {page_title}")
        return []

    page_content = get_page_content(page)
    intro_content = get_intro_content(page_content)
    sections = extract_section_titles(page_content)
    categories = get_page_categories(page)
    images = convert_images_to_png(page)
    page_html = get_page_html(page)
    tables = extract_tables(page_html)
    # TODO - fix these links with new API call
    # reference_dict = get_all_citations(page)
    # wiki_links_dict = get_external_links_by_section(page)
    reference_dict = {}
    wiki_links_dict = {}

    table_of_contents = [
        (section, subsections) for section, subsections in sections
    ] 

    return (
        page,
        page_content,
        intro_content,
        sections,
        categories,
        images,
        tables,
        reference_dict,
        wiki_links_dict,
        table_of_contents,
    )

