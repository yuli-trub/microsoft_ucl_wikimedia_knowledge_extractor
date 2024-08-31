from scripts.wiki_crawler.navigifier import get_page_sections
import logging

# TODO: check if this is still needed annd works


def get_all_page_links(page):
    """
    Get references from a Wiki page

    Parameters
        page : MediaWikiPage
            the mediawikipage object
        url : string, optional
            custom wiki API URL

    Returns
        list
           list of tuples containing section name and in-text links from the Wikipedia
    """

    # get sections from that page
    sections = get_page_sections(page)
    section_links = []

    for section in sections:
        parsed_section_links = page.parse_section_links(section)

        filtered_links = [(section, link) for link in parsed_section_links]
        section_links.extend(filtered_links)

    return section_links


def get_references(page):
    """
    Get references from a Wiki page

    Parameters
        page : MediaWikiPage
            the mediawikipage object
        url : string, optional
            custom wiki API URL

    Returns
        dict
        dictionary of section names and their corresponding references from the Wikipedia page

    """

    sections = get_page_sections(page)
    references = {}
    for section in sections:
        try:
            section_links = page.parse_section_links(section)
            filtered_references = [
                link for link in section_links if "#cite_note" in link[1]
            ]
            if filtered_references:
                references[section] = filtered_references
        except Exception as e:
            print(f"Error retrieving references in section {section}: {e}")

    return references


def get_external_links_by_section(page):
    """
    Get external links to wikipedia pages by section from a Wiki page.

    Parameters:
        page : MediaWikiPage
            the mediawikipage object

    Returns:
        dict
            dictionary of section names and their corresponding external links
    """
    sections = get_page_sections(page)
    external_links = {}
    image_extensions = {".jpg", ".jpeg", ".png", ".svg", ".gif"}

    for section in sections:
        if section in ["References", "Citations"]:
            continue
        parsed_section_links = page.parse_section_links(section)
        filtered_links = [
            link
            for link in parsed_section_links
            if "cite_note" not in link[1]
            and not any(link[1].lower().endswith(ext) for ext in image_extensions)
        ]
        if filtered_links:
            external_links[section] = list(
                dict.fromkeys(filtered_links)
            )  # delete duplicates

    return external_links


def get_cite_note_links_by_section(page):
    """
    Get cite_note links by section from a Wiki page.

    Parameters:
        page : MediaWikiPage
            the mediawikipage object

    Returns:
        dict
            dictionary of section names and their corresponding cite_note link numbers
    """
    sections = get_page_sections(page)
    cite_note_links = {}

    for section in sections:
        if section in ["References", "Citations"]:
            continue
        logging.info(f"Processing section: {section}")
        try:
            parsed_section_links = page.parse_section_links(section)
            logging.info(f"parsed_section_links: {parsed_section_links}")

            if parsed_section_links is None:
                logging.warning(f"No links found in section: {section}")
                continue

            filtered_links = [
                link[0].strip("[]")
                for link in parsed_section_links
                if "cite_note" in link[1]
            ]
            logging.info(f"filtered_links for '{section}': {filtered_links}")

            if filtered_links:
                cite_note_links[section] = filtered_links
        except Exception as e:
            logging.error(f"Error processing section '{section}': {e}")

    return cite_note_links


def get_reference_section_links(page):
    """
    Get reference section links from a Wiki page

    Parameters
        page : MediaWikiPage
            the mediawikipage object


    Returns
        list
        list of all the references in section

    """

    try:
        reference_section_links = page.parse_section_links("References")

        if not reference_section_links:
            reference_section_links = page.parse_section_links("Citations")

    except Exception as e:
        print(f"Error retrieving references in References section: {e}")

    return reference_section_links or []


def map_references_to_tuples(references):
    """
    Map references to tuples of (cite-ref, actual-link, *archived-links).

    Parameters:
        references : list
            list of references

    Returns:
        list
            list of mapped references
    """

    filtered_references = [
        ref for ref in references if "#cite_ref-type_hint" not in ref[1]
    ]
    mapped_references = []
    current_ref = None
    current_links = []
    current_title = None

    for ref in filtered_references:
        if ref[0] == "^":
            # When we encounter a new cite-ref, save the current tuple
            if current_ref and current_links:
                mapped_references.append(
                    (current_ref, current_title, current_links[0], *current_links[1:])
                )
            # Start a new tuple
            current_ref = ref[1]
            current_links = []
            current_title = None

        elif not current_title:
            current_title = ref[0]
            current_links.append(ref[1])
        else:
            current_links.append(ref[1])

    if current_ref:
        mapped_references.append(
            (current_ref, current_title, current_links[0], *current_links[1:])
        )

    return mapped_references


def create_section_links_dict(sections_with_refs, mapped_references):
    """
    Create a dictionary with sections and their actual and archived links.

    Parameters:
        sections_with_refs : dict
            Dictionary of sections and their cite references.
        mapped_references : list
            List of mapped references (cite-ref, actual-link, *archived-links).

    Returns:
        dict
            Dictionary with sections and their actual and archived links.
    """

    section_links_dict = {
        section: {"actual_links": [], "archived_links": []}
        for section in sections_with_refs
    }
    introduction_links = {"actual_links": [], "archived_links": []}

    # lookup dictionary for mapped references using the reference number
    ref_dict = {(ref[0].split("-")[-1]): ref for ref in mapped_references}

    # Track used references
    used_refs = set()

    for section, cite_refs in sections_with_refs.items():
        for cite_ref in cite_refs:
            ref_number = cite_ref.strip("[]")
            if ref_number in ref_dict:
                ref_tuple = ref_dict[ref_number]
                section_links_dict[section]["actual_links"].append(
                    (ref_tuple[1], ref_tuple[2])
                )
                section_links_dict[section]["archived_links"].append(
                    (ref_tuple[1], ref_tuple[3:])
                )
                used_refs.add(ref_number)

    # Add references that were not used in any section to the intro section
    for ref_number, ref_tuple in ref_dict.items():
        if ref_number not in used_refs:
            introduction_links["actual_links"].append((ref_tuple[1], ref_tuple[2]))
            introduction_links["archived_links"].append((ref_tuple[1], ref_tuple[3:]))

    section_links_dict = {"Introduction": introduction_links} | section_links_dict

    return section_links_dict


def get_all_citations(page):
    sections_with_refs = get_cite_note_links_by_section(page)
    mapped_references = map_references_to_tuples(get_reference_section_links(page))
    citations = create_section_links_dict(sections_with_refs, mapped_references)
    return citations
