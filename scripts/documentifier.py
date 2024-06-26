import os
from llama_index.core import Document
from llama_index.core.schema import TextNode, ImageNode
from navigifier import (
    get_wiki_page,
    get_intro_content,
    extract_section_titles,
    get_section_content,
    get_page_content,
    sanitise_filename,
    get_page_categories,
)
from imagifier import convert_images_to_png
from summarisator import general_summarisor, get_summary
from tablifier import get_html_page, extract_tables
from referenciator import (
    get_external_links_by_section,
    get_all_citations,
)


# create document
def create_document(title, content, metadata=None):
    return Document(text=content, title=title, metadata=metadata)


# create different nodes
def create_text_node(content, metadata=None):
    return TextNode(text=content, metadata=metadata)


def create_image_node(image_data, metadata=None):
    return ImageNode(image=image_data, metadata=metadata)


def create_table_node(table_data, metadata=None):
    return TextNode(table=table_data, metadata=metadata)


def create_reference_node(link, metadata=None):
    return TextNode(text=link, metadata=metadata)


def create_citation_node(link, metadata=None):
    return TextNode(text=link, metadata=metadata)


def process_page_into_doc_and_nodes(page_title):
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

    # create summary for the whole page
    document_summary_prompt = (
        "Summarize the following Wikipedia page content. "
        "The summary should cover the main topics, key points, and significant details. "
        "The goal is to provide a comprehensive overview that captures the essence of the page, "
        "making it easy to understand the main ideas without reading the entire content.\n\n"
    )
    # document_summary = get_summary(
    #     general_summarisor(document_summary_prompt, text=page_content)
    # )
    # or
    # document_summary =  page.summary

    document_summary = "Summary of the page"
    print(f"Summary: {document_summary}")

    # Create main document for the page
    page_metadata = {
        "id": sanitise_filename(page_title),
        "title": page_title,
        "type": "page",
        "summary": document_summary,
        "categories": categories,  # not sure if a list is ok?
    }
    main_document = create_document(
        title=page_title, content=page_content, metadata=page_metadata
    )

    nodes = []
    prev_section_node = None
    prev_subsection_node = None
    prev_image_node = None
    prev_table_node = None
    prev_reference_node = None
    prev_citation_node = None

    # add text node with prev and next metadata
    def add_text_node(node, is_section=True):
        nonlocal prev_section_node, prev_subsection_node
        if is_section:
            if prev_section_node:
                prev_section_node.metadata["next"] = node.metadata["id"]
                node.metadata["prev"] = prev_section_node.metadata["id"]
            prev_section_node = node
        else:
            if prev_subsection_node:
                prev_subsection_node.metadata["next"] = node.metadata["id"]
                node.metadata["prev"] = prev_subsection_node.metadata["id"]
            prev_subsection_node = node
        nodes.append(node)

    # add image node with prev and next metadata
    def add_image_node(node):
        nonlocal prev_image_node
        if prev_image_node:
            prev_image_node.metadata["next"] = node.metadata["id"]
            node.metadata["prev"] = prev_image_node.metadata["id"]
        prev_image_node = node
        nodes.append(node)

    # add table node with prev and next metadata
    def add_table_node(node):
        nonlocal prev_table_node
        if prev_table_node:
            prev_table_node.metadata["next"] = node.metadata["id"]
            node.metadata["prev"] = prev_table_node.metadata["id"]
        prev_table_node = node
        nodes.append(node)

    def add_reference_node(node):
        nonlocal prev_reference_node
        if prev_reference_node:
            prev_reference_node.metadata["next"] = node.metadata["id"]
            node.metadata["prev"] = prev_reference_node.metadata["id"]
        prev_reference_node = node
        nodes.append(node)

    def add_citation_node(node):
        nonlocal prev_citation_node
        if prev_citation_node:
            prev_citation_node.metadata["next"] = node.metadata["id"]
            node.metadata["prev"] = prev_citation_node.metadata["id"]
        prev_citation_node = node
        nodes.append(node)

    # Create intro node
    intro_content = get_intro_content(page_content)
    if intro_content:
        intro_metadata = {
            "id": f"{page_metadata['id']}_intro",
            "parent_id": page_metadata["id"],
            "type": "section",
            "source": page_metadata["id"],
            "context_summary": document_summary,
        }
        intro_node = create_text_node(content=intro_content, metadata=intro_metadata)
        add_text_node(intro_node, is_section=True)

    # Create section and subsection nodes
    sections = extract_section_titles(page_content)
    for section_title, subsections in sections:
        section_content = get_section_content(page, section_title)
        if section_content:
            section_metadata = {
                "id": f"{sanitise_filename(section_title)}",
                "parent_id": page_metadata["id"],
                "source": page_metadata["id"],
                "type": "section",
                "context_summary": document_summary,
            }
            section_node = create_text_node(
                content=section_content, metadata=section_metadata
            )
            add_text_node(section_node, is_section=True)

            prev_subsection_node = None

            for subsection_title in subsections:
                subsection_content = get_section_content(page, subsection_title)
                if subsection_content:
                    subsection_metadata = {
                        "id": f"{sanitise_filename(subsection_title)}",
                        "parent_id": section_metadata["id"],
                        "source": page_metadata["id"],
                        "type": "subsection",
                    }
                    subsection_node = create_text_node(
                        content=subsection_content, metadata=subsection_metadata
                    )
                    add_text_node(subsection_node, is_section=False)

    # create image nodes - todo: later add classification and metadata type of plot or image
    for image in images:
        image_metadata = {
            "id": image["image_name"],
            "type": "image",
            "source": page_metadata["id"],
            "parent_id": page_metadata["id"],
            "context_summary": document_summary,
        }
        image_node = create_image_node(
            image_data=image["image_data"], metadata=image_metadata
        )
        add_image_node(image_node)

    # create table nodes
    for idx, table in enumerate(tables):
        table_content = table.to_csv(index=False)
        table_metadata = {
            "id": f"{page_metadata['id']}_table_{idx}",
            "parent_id": page_metadata["id"],
            "source": page_metadata["id"],
            "type": "table",
            "context_summary": document_summary,
        }
        table_node = create_text_node(content=table_content, metadata=table_metadata)
        add_table_node(table_node)

    # create wiki refs nodes
    for section, links in wiki_links_dict.items():
        for link in links:
            reference_metadata = {
                "id": f"{sanitise_filename(link[0])}",
                "parent_id": sanitise_filename(section),
                "source": page_metadata["id"],
                "type": "wiki-ref",
            }
            reference_node = create_reference_node(link[1], metadata=reference_metadata)
            add_reference_node(reference_node)

    # create citation nodes
    for section, links in reference_dict.items():
        for link in links["actual_links"]:
            url = link[1]
            title = sanitise_filename(link[0].strip('""'))
            citation_metadata = {
                "id": title,
                "parent_id": sanitise_filename(section),
                "source": page_metadata["id"],
                "type": "citation",
            }
            citation_node = create_citation_node(url, metadata=citation_metadata)
            add_citation_node(citation_node)

            # archived links nodes
        for idx, urls in enumerate(links["archived_links"]):
            title = sanitise_filename(urls[0].strip('""'))
            if len(urls[1]) == 0:
                continue
            for url_idx, url in enumerate(urls[1]):
                unique_suffix = (
                    f"archive-{url_idx + 1}" if len(urls[1]) > 1 else "archive"
                )
                citation_metadata = {
                    "id": f"{title}-{unique_suffix}",
                    "parent_id": sanitise_filename(section),
                    "source": page_metadata["id"],
                    "type": "archive-citation",
                }
                citation_node = create_citation_node(url, metadata=citation_metadata)
                add_citation_node(citation_node)

    return [main_document] + nodes


base_dir = "../data"
documents = process_page_into_doc_and_nodes("Python (programming language)")


def check_nodes(documents, type):
    for doc in documents:
        if "type" in doc.metadata and doc.metadata["type"] == type:
            print(f"Node ID: {doc.metadata['id']}")
            print(f"Node Content: {doc.text}")
            print()


check_nodes(documents, "archive-citation")

# for doc in documents:
#     print(f"metadata:{doc.metadata}")

#     print(f"Content: {doc.text[:100]}...")
#     print()
