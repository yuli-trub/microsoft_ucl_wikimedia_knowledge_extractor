import sys
import os

# Add the project root to sys.path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from scripts.wiki_crawler.navigifier import (
    get_section_content
)

from scripts.llama_ingestionator.image_classifier import classify_and_update_image_type

# from transformator import general_summarisor, get_summary
from scripts.wiki_crawler.data_fetcher import fetch_wiki_data
from scripts.llama_ingestionator.node_creator import (
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
import logging
from scripts.helper import log_duration, sanitise_filename


# create main doc
def create_main_document(page_title, page_content, document_summary, categories):
    """Create the main llamaindex document representing the Wikimedia page
    
    Args:
        page_title (str): the title of the Wikimedia page
        page_content (str): the content of the Wikimedia page
        document_summary (str): the summary of the Wikimedia page
        categories (list): the categories of the Wikimedia page

    Returns:
        Document: LlamaIndex object -  document representing the Wikimedia page
    """
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
    """ Create Llamaindex nodes for sections and subsections of the wikimedia page
    
    Args:   
        sections (list): list of tuples containing section title and list of subsections
        main_document (Document): the main document representing the Wikimedia page
        document_summary (str): the summary of the Wikimedia page
        page (MediaWikiPage): the MediaWikiPage object
    
    Returns:
        list: list of LlamaIndex nodes representing the sections and subsections
        dict: dictionary mapping section titles to their node IDs
    """
    logging.info(f"Processing sections for main document: {main_document.doc_id}")
    nodes = []
    prev_section_node = None
    prev_subsection_node = None
    section_node_map = {}

    # Process sections
    for section_title, subsections in sections:
        section_content = get_section_content(page, section_title)
        if section_content:
            section_metadata = {
                "title": f"{sanitise_filename(section_title)}",
                "source_page": main_document.metadata["title"],
                "type": "section",
                "context": document_summary,
            }
            section_node = create_text_node(
                    content=section_content, 
                    metadata=section_metadata,
                    parent_id=main_document.doc_id,
                    source_id=main_document.doc_id
                )
            prev_section_node = add_text_node(
                nodes, section_node, prev_section_node, is_section=True
            )

            section_node_map[section_title] = section_node.node_id

            prev_subsection_node = None

            # Process subsections
            for subsection_title in subsections:
                subsection_content = get_section_content(page, subsection_title)
                if subsection_content:
                    subsection_metadata = {
                        "title": f"{sanitise_filename(subsection_title)}",
                        "source_page": main_document.metadata["title"],
                        "type": "subsection",
                        "context": document_summary,
                    }
                    subsection_node = create_text_node(
                        content=subsection_content, 
                        metadata=subsection_metadata,
                        parent_id=section_node.node_id,
                        source_id=main_document.doc_id
                    )
                    prev_subsection_node = add_text_node(
                        nodes, subsection_node, prev_subsection_node, is_section=False
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
        image_type = classify_and_update_image_type(image["image_data"], image['image_name'])
        if "image" in image_type:
            image_type = "image"

        image_metadata = {
            "title": image["image_name"],
            "type": image_type,
            "source_page": main_document.metadata["title"],
            "context": document_summary,
            "url": image["image_url"],
        }
        image_node = create_image_node(
            image_data=image["image_data"], 
            metadata=image_metadata,
            parent_id=main_document.doc_id,
            source_id=main_document.doc_id
        )
        prev_image_node = add_image_node(nodes, image_node, prev_image_node)
       
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
            "source_page": main_document.metadata["title"],
            "type": "table",
            "context": document_summary,
        }
        table_node = create_table_node(
            table_data=table_content, 
            metadata=table_metadata, 
            parent_id=main_document.doc_id,
            source_id=main_document.doc_id
        )
        prev_table_node = add_table_node(nodes, table_node, prev_table_node)
       
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
                "parent_section": sanitise_filename(section),
                "source_page": main_document.metadata["title"],
                "type": "citation",
                "context": document_summary,
            }
            citation_node = create_citation_node(
                url, 
                metadata=citation_metadata,
                parent_id=parent_node_id, 
                source_id=main_document.doc_id
            )
            prev_citation_node = add_citation_node(
                nodes, citation_node, prev_citation_node
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
                    "parent_section": sanitise_filename(section),
                    "source_page": main_document.metadata["title"],
                    "type": "archive-citation",
                }
                citation_node = create_citation_node(
                    url,
                    metadata=citation_metadata,
                    parent_id=parent_node_id, 
                    source_id=main_document.doc_id
                )   
                prev_citation_node = add_citation_node(
                    nodes, citation_node, prev_citation_node
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
                "parent_section": sanitise_filename(section),
                "source_page": main_document.metadata["title"],
                "type": "wiki-ref",
                "context": document_summary,
            }
            reference_node = create_reference_node(
                link[1],
                metadata=reference_metadata, 
                parent_id=parent_node_id, 
                source_id=main_document.doc_id
            )
            prev_reference_node = add_reference_node(
                nodes, reference_node, prev_reference_node
            )
    logging.info("Wiki links processed successfully")
    return nodes


# create all the nodes - returns clean initial loaded nodes
@log_duration
def process_page_into_doc_and_nodes(page_title):
    """Process a Wikipedia page to create LlamaIndex nodes

    Args:
        page_title (str): the title of the Wikipedia page

    Returns:
        list: list of LlamaIndex nodes representing the Wikipedia page
    """

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

    logging.info("Wiki data fetched successfully")

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
            "type": "section",
            "context": document_summary,
        }
        intro_node = create_text_node(
            content=intro_content, 
            metadata=intro_metadata,
            parent_id=main_document.doc_id, 
            source_id=main_document.doc_id
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

