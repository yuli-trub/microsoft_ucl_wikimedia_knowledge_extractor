from mediawiki import MediaWiki
from scripts.navigifier import get_page_sections, get_section_content


def get_all_page_links(page):
    """
    Get references from a Wiki page

    Parameters
        page : MediaWikiPage
            the mediawikipage object
        url : string, optional
            custom wiki API URL

    Returns
        list
            list of in text links usually to other pages from the Wikipedia page
    """
    # wikipedia = MediaWiki(user_agent="KnowledgeExtractor/1.0 (ucabytr@ucl.ac.uk)")

    # if url:
    #     try:
    #         wikipedia.set_api_url(url)
    #     except Exception as e:
    #         print(f"Error setting API URL: {e}. Defaulting to Wikipedia.")
    #         wikipedia.set_api_url("https://en.wikipedia.org/w/api.php")

    # # initialise the wikimediapage object
    # page = wikipedia.page(page_title)

    # get sections from that page
    sections = get_page_sections(page)

    # initialise list of in text links
    page_links = []
    for section in sections:
        parsed_section_links = page.parse_section_links(section)

        # filter out image links and footnotes?
        # links to other wiki content?
        filtered_links = [
            link
            for link in parsed_section_links
            if "File:" not in link[1] and "#cite_note" not in link[1]
        ]
        page_links.extend(filtered_links)

    return page_links


def get_references(page):
    """
    Get references from a Wiki page

    Parameters
        page : MediaWikiPage
            the mediawikipage object
        url : string, optional
            custom wiki API URL

    Returns
        list
            list of references from the Wikipedia page
    """
    # wikipedia = MediaWiki(user_agent="KnowledgeExtractor/1.0 (ucabytr@ucl.ac.uk)")

    # if url:
    #     try:
    #         wikipedia.set_api_url(url)
    #     except Exception as e:
    #         print(f"Error setting API URL: {e}. Defaulting to Wikipedia.")
    #         wikipedia.set_api_url("https://en.wikipedia.org/w/api.php")

    # page = wikipedia.page(page_title)

    sections = get_page_sections(page)
    # it should include it I think - but extra check?
    if "References" in sections:
        try:
            references = page.parse_section_links("References")

            # filter put the self-ref back to the page content
            filtered_references = [
                reference for reference in references if "#cite_ref" not in reference[1]
            ]

            return filtered_references
        except Exception as e:
            print(f"Error retrieving references: {e}")
            return []


get_references("Python (programming language)")
