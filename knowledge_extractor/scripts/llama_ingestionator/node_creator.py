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


# Create different type nodes
def create_text_node(content, metadata=None, parent_id=None, source_id=None):
    node = TextNode(text=content, metadata=metadata)
    if parent_id:
        node.relationships[NodeRelationship.PARENT] = RelatedNodeInfo(node_id=parent_id)
    if source_id:
        node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=source_id)
    return node

def create_image_node(image_data, metadata=None, parent_id=None, source_id=None):
    node = ImageNode(image=image_data, metadata=metadata)
    if parent_id:
        node.relationships[NodeRelationship.PARENT] = RelatedNodeInfo(node_id=parent_id)
    if source_id:
        node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=source_id)
    return node


def create_table_node(table_data, metadata=None, parent_id=None, source_id=None):
    node = TextNode(text=table_data, metadata=metadata)
    if parent_id:
        node.relationships[NodeRelationship.PARENT] = RelatedNodeInfo(node_id=parent_id)
    if source_id:
        node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=source_id)
    return node


def create_reference_node(link, metadata=None, parent_id=None, source_id=None):
    node = ImageNode(text=link, metadata=metadata)
    if parent_id:
        node.relationships[NodeRelationship.PARENT] = RelatedNodeInfo(node_id=parent_id)
    if source_id:
        node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=source_id)
    return node


def create_citation_node(link, metadata=None, parent_id=None, source_id=None):
    node = ImageNode(text=link, metadata=metadata)
    if parent_id:
        node.relationships[NodeRelationship.PARENT] = RelatedNodeInfo(node_id=parent_id)
    if source_id:
        node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=source_id)
    return node


# Add nodes with prev and next metadata
def add_text_node(nodes, node, prev_node, is_section=True):
    """ Add text node to the list of nodes with prev and next metadata
    
    Args:
        nodes (list): list of nodes
        node (Node): node to be added
        prev_node (Node): previous node
        is_section (bool): whether the node is a section or not 

    Returns:
        Node: the previous node
    """
    if is_section:
        if prev_node:
            prev_node.relationships[NodeRelationship.NEXT] = RelatedNodeInfo(
                node_id=node.node_id
            )
            node.relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(
                node_id=prev_node.node_id
            )
        prev_node = node
    else:
        if prev_node:
            prev_node.relationships[NodeRelationship.NEXT] = RelatedNodeInfo(
                node_id=node.node_id
            )
            node.relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(
                node_id=prev_node.node_id
            )
        prev_node = node
    nodes.append(node)
    return prev_node


def add_image_node(nodes, node, prev_node):
    """ Add image node to the list of nodes with prev and next metadata

    Args:
        nodes (list): list of nodes
        node (Node): node to be added
        prev_node (Node): previous node
    
    Returns:
        Node: the previous node
    """
    if prev_node:
        prev_node.relationships[NodeRelationship.NEXT] = RelatedNodeInfo(
            node_id=node.node_id
        )
        node.relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(
            node_id=prev_node.node_id
        )
    prev_node = node
    nodes.append(node)
    return prev_node


def add_table_node(nodes, node, prev_node):
    """ Add table node to the list of nodes with prev and next metadata

    Args:
        nodes (list): list of nodes
        node (Node): node to be added
        prev_node (Node): previous node

    Returns:
        Node: the previous node
    """
    if prev_node:
        prev_node.relationships[NodeRelationship.NEXT] = RelatedNodeInfo(
            node_id=node.node_id
        )
        node.relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(
            node_id=prev_node.node_id
        )
    prev_node = node
    nodes.append(node)
    return prev_node


def add_reference_node(nodes, node, prev_node):
    """ Add reference node to the list of nodes with prev and next metadata

    Args:
        nodes (list): list of nodes
        node (Node): node to be added
        prev_node (Node): previous node
    
    Returns:
        Node: the previous node
    """ 
    if prev_node:
        prev_node.relationships[NodeRelationship.NEXT] = RelatedNodeInfo(
            node_id=node.node_id
        )
        node.relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(
            node_id=prev_node.node_id
        )
    prev_node = node
    nodes.append(node)
    return prev_node


def add_citation_node(nodes, node, prev_node):
    """ Add citation node to the list of nodes with prev and next metadata

    Args:
        nodes (list): list of nodes
        node (Node): node to be added
        prev_node (Node): previous node

    Returns:
        Node: the previous node
    """
    if prev_node:
        prev_node.relationships[NodeRelationship.NEXT] = RelatedNodeInfo(
            node_id=node.node_id
        )
        node.relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(
            node_id=prev_node.node_id
        )
    prev_node = node
    nodes.append(node)
    return prev_node
