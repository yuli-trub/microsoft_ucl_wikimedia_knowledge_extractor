import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from scripts.wiki_crawler.navigifier import (
    get_section_content,
    sanitise_filename,
)

from llama_ingestionator.image_classifier import classify_and_update_image_type

# from transformator import general_summarisor, get_summary
from scripts.wiki_crawler.data_fetcher import fetch_wiki_data
from llama_ingestionator.node_creator import (
    create_document,
    create_text_node,
    create_image_node,
    create_table_node,
    create_reference_node,
    create_citation_node,
    add_text_node,
    add_image_node,
    add_table_node,
    add_reference_node,
    add_citation_node,
)
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo
import logging
from helper import log_duration


# create main doc
def create_main_document(page_title, page_content, document_summary, categories):
    page_metadata = {
        "title": sanitise_filename(page_title),
        "type": "page",
        "summary": document_summary,
        "categories": categories,
    }

    document = create_document(
        title=page_title, content=page_content, metadata=page_metadata
    )
    logging.info(f"Main document created with ID: {document.doc_id}")
    return document


# create section and subsection nodes
def process_sections(sections, main_document, document_summary, page):
    logging.info(f"Processing sections for main document: {main_document.doc_id}")
    nodes = []
    prev_section_node = None
    prev_subsection_node = None
    section_node_map = {}

    for section_title, subsections in sections:
        section_content = get_section_content(page, section_title)
        if section_content:
            section_metadata = {
                "title": f"{sanitise_filename(section_title)}",
                "source": main_document.doc_id,
                "type": "section",
                "context": document_summary,
            }
            section_node = create_text_node(
                content=section_content, metadata=section_metadata
            )
            prev_section_node = add_text_node(
                nodes, section_node, prev_section_node, is_section=True
            )
            section_node.relationships[NodeRelationship.PARENT] = RelatedNodeInfo(
                node_id=main_document.doc_id
            )
            section_node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(
                node_id=main_document.doc_id
            )

            section_node_map[section_title] = section_node.node_id

            prev_subsection_node = None
            for subsection_title in subsections:
                subsection_content = get_section_content(page, subsection_title)
                if subsection_content:
                    subsection_metadata = {
                        "title": f"{sanitise_filename(subsection_title)}",
                        "source": main_document.doc_id,
                        "type": "subsection",
                        "context": document_summary,
                    }
                    subsection_node = create_text_node(
                        content=subsection_content, metadata=subsection_metadata
                    )
                    prev_subsection_node = add_text_node(
                        nodes, subsection_node, prev_subsection_node, is_section=False
                    )
                    subsection_node.relationships[NodeRelationship.PARENT] = (
                        RelatedNodeInfo(node_id=section_node.node_id)
                    )
                    section_node_map[subsection_title] = section_node.node_id

    logging.info("Sections processed successfully")
    return nodes, section_node_map


# create image nodes
def process_images(images, main_document, document_summary):
    logging.info(f"Processing images for main document: {main_document.doc_id}")
    nodes = []
    prev_image_node = None
    for image in images:
        logging.info(f"Processing image - classifying: {image['image_name']}")
        image_type = classify_and_update_image_type(image["image_data"])
        if "image" in image_type:
            image_type = "image"
        logging.info(f"Image type: {image_type}")

        image_metadata = {
            "title": image["image_name"],
            "type": image_type,
            "source": main_document.doc_id,
            "context": document_summary,
            "url": image["image_url"],
        }
        image_node = create_image_node(
            image_data=image["image_data"], metadata=image_metadata
        )
        prev_image_node = add_image_node(nodes, image_node, prev_image_node)
        image_node.relationships[NodeRelationship.PARENT] = RelatedNodeInfo(
            node_id=main_document.doc_id
        )
        image_node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(
            node_id=main_document.doc_id
        )
    logging.info("Images processed successfully")
    return nodes


# create table nodes
def process_tables(tables, main_document, document_summary):
    logging.info(f"Processing tables for main document: {main_document.doc_id}")
    nodes = []
    prev_table_node = None
    for idx, table in enumerate(tables):
        table_content = table.to_csv(index=False)
        table_metadata = {
            "title": f"{main_document.metadata['title']}_table_{idx}",
            "source": main_document.doc_id,
            "type": "table",
            "context": document_summary,
        }
        table_node = create_table_node(
            table_data=table_content, metadata=table_metadata
        )
        prev_table_node = add_table_node(nodes, table_node, prev_table_node)
        table_node.relationships[NodeRelationship.PARENT] = RelatedNodeInfo(
            node_id=main_document.doc_id
        )
        table_node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(
            node_id=main_document.doc_id
        )
    logging.info("Tables processed successfully")
    return nodes


# create citation nodes
def process_references(
    reference_dict, main_document, document_summary, section_node_map
):
    logging.info(f"Processing references for main document: {main_document.doc_id}")
    nodes = []
    prev_citation_node = None
    for section, links in reference_dict.items():
        parent_node_id = section_node_map.get(section)
        for link in links["actual_links"]:
            url = link[1]
            title = sanitise_filename(link[0].strip('""'))
            citation_metadata = {
                "title": title,
                "parent_id": sanitise_filename(section),
                "source": main_document.doc_id,
                "type": "citation",
                "context": document_summary,
            }
            citation_node = create_citation_node(url, metadata=citation_metadata)
            prev_citation_node = add_citation_node(
                nodes, citation_node, prev_citation_node
            )
            citation_node.relationships[NodeRelationship.PARENT] = RelatedNodeInfo(
                node_id=parent_node_id
            )
            citation_node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(
                node_id=main_document.doc_id
            )

        for idx, urls in enumerate(links["archived_links"]):
            title = sanitise_filename(urls[0].strip('""'))
            if len(urls[1]) == 0:
                continue
            for url_idx, url in enumerate(urls[1]):
                unique_suffix = (
                    f"archive-{url_idx + 1}" if len(urls[1]) > 1 else "archive"
                )
                citation_metadata = {
                    "title": f"{title}-{unique_suffix}",
                    "parent_id": sanitise_filename(section),
                    "source": main_document.doc_id,
                    "type": "archive-citation",
                }
                citation_node = create_citation_node(url, metadata=citation_metadata)
                prev_citation_node = add_citation_node(
                    nodes, citation_node, prev_citation_node
                )
                citation_node.relationships[NodeRelationship.PARENT] = RelatedNodeInfo(
                    node_id=parent_node_id
                )
                citation_node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(
                    node_id=main_document.doc_id
                )
    logging.info("References processed successfully")
    return nodes


# create wiki link nodes
def process_wiki_links(
    wiki_links_dict, main_document, document_summary, section_node_map
):
    logging.info(f"Processing wiki links for main document: {main_document.doc_id}")
    nodes = []

    prev_reference_node = None
    for section, links in wiki_links_dict.items():
        parent_node_id = section_node_map.get(section)
        for link in links:
            reference_metadata = {
                "title": f"{sanitise_filename(link[0])}",
                "parent_id": sanitise_filename(section),
                "source": main_document.doc_id,
                "type": "wiki-ref",
                "context": document_summary,
            }
            reference_node = create_reference_node(link[1], metadata=reference_metadata)
            prev_reference_node = add_reference_node(
                nodes, reference_node, prev_reference_node
            )
            reference_node.relationships[NodeRelationship.PARENT] = RelatedNodeInfo(
                node_id=parent_node_id
            )
            reference_node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(
                node_id=main_document.doc_id
            )
    logging.info("Wiki links processed successfully")
    return nodes


# create all the nodes - returns clean initial loaded nodes
@log_duration
def process_page_into_doc_and_nodes(page_title):

    # fetch wiki data
    (
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
    ) = fetch_wiki_data(page_title)

    logging.debug("Wiki data fetched successfully")
    # create summary for the whole page
    # document_summary_prompt = (
    #     "Summarize the following Wikipedia page content. "
    #     "The summary should cover the main topics, key points, and significant details. "
    #     "The goal is to provide a comprehensive overview that captures the essence of the page, "
    #     "making it easy to understand the main ideas without reading the entire content.\n\n"
    # )
    # document_summary = get_summary(
    #     general_summarisor(document_summary_prompt, text=page_content)
    # )
    # or
    logging.info(f"Creating main document for page: {page_title}")
    document_summary = page.summary

    main_document = create_main_document(
        page_title, page_content, document_summary, categories
    )
    nodes = [main_document]

    prev_section_node = None

    # Create intro node
    if intro_content:
        intro_metadata = {
            "title": f"{main_document.metadata['title']}_intro",
            "parent_id": main_document.doc_id,
            "type": "section",
            "source": main_document.doc_id,
            "context": document_summary,
        }
        intro_node = create_text_node(content=intro_content, metadata=intro_metadata)
        intro_node.relationships[NodeRelationship.PARENT] = RelatedNodeInfo(
            node_id=main_document.doc_id
        )
        prev_section_node = add_text_node(
            nodes, intro_node, prev_section_node, is_section=True
        )

    # Process sections, images, tables, references, and wiki links
    section_nodes, section_node_map = process_sections(
        sections, main_document, document_summary, page
    )
    image_nodes = process_images(images, main_document, document_summary)
    table_nodes = process_tables(tables, main_document, document_summary)
    reference_nodes = process_references(
        reference_dict, main_document, document_summary, section_node_map
    )
    wiki_link_nodes = process_wiki_links(
        wiki_links_dict, main_document, document_summary, section_node_map
    )

    all_nodes = (
        nodes
        + section_nodes
        + image_nodes
        + table_nodes
        + reference_nodes
        + wiki_link_nodes
    )
    logging.info("All nodes processed successfully")
    logging.info(f"Total nodes created: {len(all_nodes)}")
    return all_nodes


# bit to test out the node creation
# documents = process_page_into_doc_and_nodes("Python (programming language)")


def check_nodes(documents, type):
    for doc in documents:
        if "type" in doc.metadata and doc.metadata["type"] == type:
            print(f"Node ID: {doc.node_id}")
            print(f"Node title: {doc.metadata['title']}")

            print(f"Node meta: {doc.relationships}")

            # print(f"Node Content: {doc.text}")  # doc.image or doc.text
            print()


# check_nodes(documents, "section")
