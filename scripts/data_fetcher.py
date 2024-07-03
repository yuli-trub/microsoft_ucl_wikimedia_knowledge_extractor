from navigifier import (
    get_wiki_page,
    get_intro_content,
    extract_section_titles,
    get_page_content,
    get_page_categories,
)
from tablifier import get_html_page, extract_tables
from referenciator import (
    get_external_links_by_section,
    get_all_citations,
)
from imagifier import convert_images_to_png


# helper function to fetch all of the cleaned data from the wiki page in a structured format
def fetch_wiki_data(page_title):
    page = get_wiki_page(page_title)
    if not page:
        print(f"Failed to retrieve the page: {page_title}")
        return []

    page_content = get_page_content(page)
    intro_content = get_intro_content(page_content)
    sections = extract_section_titles(page_content)
    categories = get_page_categories(page)
    images = convert_images_to_png(page)
    page_html = get_html_page(page)
    tables = extract_tables(page_html)
    reference_dict = get_all_citations(page)
    wiki_links_dict = get_external_links_by_section(page)

    table_of_contents = [
        (section, subsections) for section, subsections in sections
    ]  # add to metadata to provide content?
    # table_of_contents = page.table_of_contents

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