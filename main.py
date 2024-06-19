from scripts.searchinator import search_wiki
from scripts.navigifier import (
    get_wiki_page,
    get_page_sections,
    get_section_content,
    process_and_save_sections,
    sanitise_filename,
)
from scripts.referenciator import get_all_page_links, get_references
from scripts.imagifier import convert_images_to_png
from scripts.tablifier import extract_tables
from mediawiki import MediaWiki
import os


def main():
    wikipedia = MediaWiki(user_agent="KnowledgeExtractor/1.0 (ucabytr@ucl.ac.uk)")

    # searchinator - search for pages
    query = "Spider-Man"
    # url = "https://marvels.fandom.com/api.php"
    url = "https://en.wikipedia.org/w/api.php"

    search_results = search_wiki(query, url)
    print("Search Results:", search_results)

    if search_results:
        for page_title in search_results:
            print(f"Selected Page: {page_title}")
            base_dir = f"./data"

            clean_page_name = sanitise_filename(page_title)
            page_dir = f"{base_dir}/{clean_page_name}"
            print(f"Page Directory: {page_dir}")
            os.makedirs(page_dir, exist_ok=True)

            # navigifier
            page = get_wiki_page(page_title)
            if not page:
                print(f"Failed to retrieve the page: {page_title}")
                continue

            process_and_save_sections(page, f"{base_dir}/{clean_page_name}/content")

            # imagifier
            # convert_images_to_png(page, f"{base_dir}/{clean_page_name}/images")

            # tablifier - to do : save in temp folder?
            # extract_tables(
            #     page_title,
            #     output_dir=f"{base_dir}/tables/{page_title.replace(' ', '_')}",
            # )

            # # referenciator - to do: save to the right place?
            # all_page_links = get_all_page_links(page)
            # print("All page links:", all_page_links)

            # references = get_references(page)
            # print("References:", references)

        else:
            print(f"Failed to retrieve the page: {page_title}")
    else:
        print(f"No search results found for query: {query}")


if __name__ == "__main__":
    main()
