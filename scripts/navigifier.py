from mediawiki import MediaWiki
import re
import os

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


# splitting the sectins and subsections
def extract_section_titles(page_content):
    # identify pattern with == section == or === subsection ===
    section_pattern = re.compile(r"(^==[^=].*?==)|(^===.*?===)", re.MULTILINE)

    # find all
    titles = section_pattern.findall(page_content)

    # clean up
    sections = []
    for title in titles:
        section_title = title[0].strip("= ").strip() if title[0] else None
        subsection_title = title[1].strip("= ").strip() if title[1] else None
        if section_title:
            sections.append((section_title, "section"))
        elif subsection_title:
            sections.append((subsection_title, "subsection"))

    return sections


def sanitise_filename(filename):
    """
    Sanitise a string to be used as a filename

    Parameters
        filename : str
            The string to sanitise

    Returns
            safe_filename : str
                The sanitised filename
    """
    return re.sub(r'[\\/*?:"<>|]', "_", filename)


# get content for each section and subsection and save it in a dictionary hierarchiecally
def save_content_to_file(base_dir, section_title, content, is_subsection=False):
    """
    Save content to a file in the specified directory

    Parameters:
        base_dir : str
            The base directory to save the content
        section_title : str
            The title of the section or subsection
        content : str
            The content to save
        is_subsection : bool, optional
            Whether the content is in a subsection. Default = False

    Returns:
        None
    """
    safe_title = sanitise_filename(section_title)
    dir_path = (
        base_dir
        if not is_subsection
        else os.path.join(base_dir, "subsections", safe_title)
    )
    os.makedirs(dir_path, exist_ok=True)
    filename = (
        "content.txt"
        if is_subsection
        else "intro.txt" if section_title == "Introduction" else "section.txt"
    )
    file_path = os.path.join(dir_path, filename)
    try:
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(f"{section_title}\n\n{content}")
        print(f"Saved content to {file_path}")
    except Exception as e:
        print(f"Failed to save content to {file_path}: {e}")


def process_and_save_sections(page, base_dir):
    os.makedirs(base_dir, exist_ok=True)
    intro_content = get_intro_content(page.content)
    if intro_content:
        print("Got the intro")
        save_content_to_file(base_dir, "Introduction", intro_content)
    sections = extract_section_titles(page.content)
    for section_title, section_type in sections:
        section_content = get_section_content(page, section_title)
        if section_content:
            section_dir = os.path.join(base_dir, section_title)
            if section_type == "section":
                print(f"Processing section: {section_title}")
                save_content_to_file(section_dir, section_title, section_content)
            elif section_type == "subsection":
                print(f"Processing subsection: {section_title}")
                save_content_to_file(
                    section_dir, section_title, section_content, is_subsection=True
                )


page_content = """
== Section 1 ==
Content of section 1
=== Subsection 1.1 ===
Content of subsection 1.1
=== Subsection 1.2 ===
Content of subsection 1.2
== Section 2 ==
Content of section 2
"""

# print(get_page_html(get_wiki_page("Python (programming language)")))
# print(get_intro_content((get_page_content((get_wiki_page("Mount Everest"))))))
# print(extract_sections(get_page_content((get_wiki_page("Mount Everest")))))

process_and_save_sections((((get_wiki_page("Mount Everest")))), "../data/content")
