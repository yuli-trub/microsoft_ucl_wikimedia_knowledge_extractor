import os
from llama_index.core import Document, VectorStoreIndex
from navigifier import sanitise_filename


def load_text_documents(base_dir, page_name):
    """
    Load documents from the specified directory

    Parameters:
        base_dir : str
            The base directory to load the documents

    Returns:
        list
            List of Document objects
    """

    documents = []
    page_name = sanitise_filename(page_name)
    page_dir = os.path.join(base_dir, page_name)

    intro_path = os.path.join(page_dir, "content", "Introduction", "Introduction.txt")
    if os.path.exists(intro_path):
        with open(intro_path, "r", encoding="utf-8") as f:
            intro_content = f.read()
        intro_doc = Document(
            text=f"Introduction\n{intro_content}",
            title="Introduction",
            metadata={"id": f"{page_name}_intro", "parent_id": f"{page_name}"},
        )
        documents.append(intro_doc)

    content_dir = os.path.join(page_dir, "content")
    for section in os.listdir(content_dir):
        section_path = os.path.join(content_dir, section)
        if os.path.isdir(section_path) and section != "Introduction":
            section_id = sanitise_filename(section)
            section_content_path = os.path.join(section_path, f"{section_id}.txt")
            if os.path.exists(section_content_path):
                with open(section_content_path, "r", encoding="utf-8") as f:
                    section_content = f.read()
                section_doc = Document(
                    text=f"{section}\n{section_content}",
                    title=section,
                    metadata={
                        "id": f"{page_name}_{section_id}",
                        "parent_id": f"{page_name}",
                    },
                )
                documents.append(section_doc)

            for subsection in os.listdir(section_path):
                if subsection != f"{section_id}.txt":
                    subsection_path = os.path.join(section_path, subsection)
                    if os.path.isdir(subsection_path):
                        subsection_id = sanitise_filename(subsection)
                        subsection_content_path = os.path.join(
                            # get rid of sub if i get rid of it in navigifier?
                            subsection_path,
                            f"sub-{subsection_id}.txt",
                        )
                        if os.path.exists(subsection_content_path):
                            with open(
                                subsection_content_path, "r", encoding="utf-8"
                            ) as f:
                                subsection_content = f.read()
                            subsection_doc = Document(
                                text=f"{subsection_id}\n{subsection_content}",
                                title=subsection_id,
                                metadata={
                                    "id": f"{page_name}_{section_id}_{subsection_id}",
                                    "parent_id": f"{page_name}_{section_id}",
                                },
                            )
                            documents.append(subsection_doc)

    return documents


base_dir = "../data"
documents = load_text_documents(base_dir, "Spider-Man")

for doc in documents:
    print(f"metadata: {doc.metadata}")
    print(f"Content: {doc.text[:100]}...")
    print()
