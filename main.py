from scripts.searchinator import search_wiki
from scripts.navigifier import get_wiki_page, get_page_sections, get_section_content
from scripts.referenciator import get_all_page_links, get_references
from scripts.imagifier import convert_images_to_png
from mediawiki import MediaWiki


def main():
    wikipedia = MediaWiki(user_agent="KnowledgeExtractor/1.0 (ucabytr@ucl.ac.uk)")

    # searchinator - search for pages
    query = "Spider-Man"
    url = "https://marvels.fandom.com/api.php"
    # url = "https://en.wikipedia.org/w/api.php"

    search_results = search_wiki(query, url)
    print("Search Results:", search_results)

    if search_results:
        # choose the first page
        page_title = search_results[0]
        print("Selected Page:", page_title)

        # navigifier
        page = get_wiki_page(page_title)
        if page:
            sections = get_page_sections(page)
            print("Page Sections:", sections)

            # content for one section - to test -  will change to iteration over all sections
            if sections:
                section_title = sections[2]
                section_content = get_section_content(page, section_title)
                print(f"Content of section '{section_title}':", section_content)

            # imagifier
            convert_images_to_png(page)

            # referenciator
            all_page_links = get_all_page_links(page)
            print("All page links:", all_page_links)

            references = get_references(page)
            print("References:", references)

        else:
            print(f"Failed to retrieve the page: {page_title}")
    else:
        print(f"No search results found for query: {query}")


if __name__ == "__main__":
    main()
