import os
from llama_index.core import Document
from llama_index.core.schema import TextNode, ImageNode
from navigifier import (
    get_section_content,
    sanitise_filename,
)

# from transformator import general_summarisor, get_summary

from data_fetcher import fetch_wiki_data

from node_creator import (
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


# TODO: probably change matadata relationships moving them to relationships dict?


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
    document_summary = page.summary

    # document_summary = "Summary of the page"
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

    # Create intro node
    if intro_content:
        intro_metadata = {
            "id": f"{page_metadata['id']}_intro",
            "parent_id": page_metadata["id"],
            "type": "section",
            "source": page_metadata["id"],
            "context": document_summary,
        }
        intro_node = create_text_node(content=intro_content, metadata=intro_metadata)
        prev_section_node = add_text_node(
            nodes, intro_node, prev_section_node, is_section=True
        )

    # Create section and subsection nodes
    for section_title, subsections in sections:
        section_content = get_section_content(page, section_title)
        if section_content:
            section_metadata = {
                "id": f"{sanitise_filename(section_title)}",
                "parent_id": page_metadata["id"],
                "source": page_metadata["id"],
                "type": "section",
                "context": document_summary,
            }
            section_node = create_text_node(
                content=section_content, metadata=section_metadata
            )
            prev_section_node = add_text_node(
                nodes, section_node, prev_section_node, is_section=True
            )

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
                    prev_subsection_node = add_text_node(
                        nodes, subsection_node, prev_subsection_node, is_section=False
                    )

    # create image nodes - todo: later add classification and metadata type of plot or image
    for image in images:
        image_metadata = {
            "id": image["image_name"],
            "type": "image",
            "source": page_metadata["id"],
            "parent_id": page_metadata["id"],
            "context": document_summary,
        }
        image_node = create_image_node(
            image_data=image["image_data"], metadata=image_metadata
        )
        prev_image_node = add_image_node(nodes, image_node, prev_image_node)

    # create table nodes
    for idx, table in enumerate(tables):
        table_content = table.to_csv(index=False)
        table_metadata = {
            "id": f"{page_metadata['id']}_table_{idx}",
            "parent_id": page_metadata["id"],
            "source": page_metadata["id"],
            "type": "table",
            "context": document_summary,
        }
        table_node = create_table_node(
            table_data=table_content, metadata=table_metadata
        )
        prev_table_node = add_table_node(nodes, table_node, prev_table_node)

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
            prev_reference_node = add_reference_node(
                nodes, reference_node, prev_reference_node
            )

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
            prev_citation_node = add_citation_node(
                nodes, citation_node, prev_citation_node
            )

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
                prev_citation_node = add_citation_node(
                    nodes, citation_node, prev_citation_node
                )

    return [main_document] + nodes


documents = process_page_into_doc_and_nodes("Python (programming language)")


def check_nodes(documents, type):
    for doc in documents:
        if "type" in doc.metadata and doc.metadata["type"] == type:
            print(f"Node ID: {doc.metadata['id']}")
            print(f"Node meta: {doc.metadata}")

            print(f"Node Content: {doc.text}")  # doc.image or doc.text
            print()


check_nodes(documents, "section")
