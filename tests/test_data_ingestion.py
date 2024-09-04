import pytest
from knowledge_extractor.scripts.wiki_crawler.data_fetcher import fetch_wiki_data
from llama_index.core.schema import Document

def test_data_ingestion():
    page_title = "Eiffel Tower"

    page, page_content, intro_content, sections, categories, images, tables, ref_dict, wiki_links_dict, toc = fetch_wiki_data(page_title)
    
    # Check that data is fetched correctly
    assert page is not None
    assert isinstance(page_content, str)
    assert len(page_content) > 0
    
    # Check if intro content is fetched correctly
    assert isinstance(sections, list)
    assert len(sections) > 0
    
    # Check if categories, images, and tables are fetched correctly
    assert isinstance(images, list)
    assert isinstance(tables, list)

def test_llamaindex_document_creation():
    from scripts.llama_ingestionator.node_creator import create_document

    # Mock data
    page_title = "Test Page"
    page_content = "This is a test content"
    metadata = {"title": page_title, "type": "page"}

    # Create a document node
    document_node = create_document(title=page_title, content=page_content, metadata=metadata)
    
    # Check if the document node is created correctly
    assert isinstance(document_node, Document)
    assert document_node.metadata["title"] == page_title
    assert document_node.metadata["type"] == "page"
    assert document_node.get_content() == page_content
