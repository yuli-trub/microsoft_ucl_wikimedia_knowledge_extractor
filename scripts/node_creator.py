from llama_index.core import Document
from llama_index.core.schema import (
    TextNode,
    ImageNode,
    NodeRelationship,
    RelatedNodeInfo,
)


# Create document
def create_document(title, content, metadata=None):
    return Document(text=content, title=title, metadata=metadata)


# Create different nodes
def create_text_node(content, metadata=None):
    return TextNode(text=content, metadata=metadata)


def create_image_node(image_data, metadata=None):
    return ImageNode(image=image_data, metadata=metadata)


def create_table_node(table_data, metadata=None):
    return TextNode(text=table_data, metadata=metadata)


def create_reference_node(link, metadata=None):
    return TextNode(text=link, metadata=metadata)


def create_citation_node(link, metadata=None):
    return TextNode(text=link, metadata=metadata)


# Add nodes with prev and next metadata
def add_text_node(nodes, node, prev_node, is_section=True):
    if is_section:
        if prev_node:
            prev_node.relationships[NodeRelationship.NEXT] = RelatedNodeInfo(
                node_id=node.node_id
            )
            node.relationships[NodeRelationship.PREVIOUS] = prev_node.node_id
        prev_node = node
    else:
        if prev_node:
            prev_node.relationships[NodeRelationship.NEXT] = RelatedNodeInfo(
                node_id=node.node_id
            )
            node.relationships[NodeRelationship.PREVIOUS] = prev_node.node_id
        prev_node = node
    nodes.append(node)
    return prev_node


def add_image_node(nodes, node, prev_node):
    if prev_node:
        prev_node.relationships[NodeRelationship.NEXT] = RelatedNodeInfo(
            node_id=node.node_id
        )
        node.relationships[NodeRelationship.PREVIOUS] = prev_node.node_id
    prev_node = node
    nodes.append(node)
    return prev_node


def add_table_node(nodes, node, prev_node):
    if prev_node:
        prev_node.relationships[NodeRelationship.NEXT] = RelatedNodeInfo(
            node_id=node.node_id
        )
        node.relationships[NodeRelationship.PREVIOUS] = prev_node.node_id
    prev_node = node
    nodes.append(node)
    return prev_node


def add_reference_node(nodes, node, prev_node):
    if prev_node:
        prev_node.relationships[NodeRelationship.NEXT] = RelatedNodeInfo(
            node_id=node.node_id
        )
        node.relationships[NodeRelationship.PREVIOUS] = prev_node.node_id
    prev_node = node
    nodes.append(node)
    return prev_node


def add_citation_node(nodes, node, prev_node):
    if prev_node:
        prev_node.relationships[NodeRelationship.NEXT] = RelatedNodeInfo(
            node_id=node.node_id
        )
        node.relationships[NodeRelationship.PREVIOUS] = prev_node.node_id
    prev_node = node
    nodes.append(node)
    return prev_node
