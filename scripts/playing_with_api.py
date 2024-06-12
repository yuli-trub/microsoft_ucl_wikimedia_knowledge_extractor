# setting up
from mediawiki import MediaWiki

# set user agent string
wikipedia = MediaWiki(user_agent="KnowledgeExtractor/1.0 (ucabytr@ucl.ac.uk)")

# set up the api url for wikias
# wikipedia.set_api_url('http://awoiaf.westeros.org/api.php')


# search by name and get list of possible related pages - contextual search
# suggestion - cool for typos
# returns tuple or list (without suggestion)
search_results = wikipedia.search("chrlos dickens", results=5, suggestion=True)
# ['Washington', 'Washington, D.C.', 'List of Governors of Washington']

# categories at the bottom of wiki page - similar pages?
categories = wikipedia.categorymembers(
    "Python (programming language)", results=5, subcategories=True
)

# initialising page to get links titles and that's it?
page = wikipedia.page(
    title="Python (programming language)",
    pageid=None,
    auto_suggest=True,
    redirect=True,
    preload=False,
)
# parse links within section - gets images, urls as tuple (title |string, url |string), footnotes also as tuple and link to the footnote on the bottom of the page, works with the subsection titles as well!
parse_links = page.parse_section_links("Indentation")

# parses section content - text
section = page.section("Indentation")


# returns text
# sections are denoted by == Section == and ===Subsection===
# excludes the code snippets, bullet pointed lists and tables
page_content = page.content


# returns html of full page
# tables are with class "wikitable"
# <table class="wikitable" style="text-align:center">
# <caption style="white-space:nowrap">Multiplication table
# </caption>
# <tbody><tr>
# <th>×</th>
# <th>1</th>
# <th>2</th>
# <th>3
# </th></tr>
# <tr>
# <th>1
# </th>
# <td>1</td>
# <td>2</td>
# <td>3
# </td></tr>
# <tr>
# <th>2
# </th>
# <td>2</td>
# <td>4</td>
# <td>6
# </td></tr>
# <tr>
# <th>3
# </th>
# <td>3</td>
# <td>6</td>
# <td>9
# </td></tr></tbody></table>
# slow fo large pages
page_html = page.html

# all the images links
images = page.images


# list of all sections
sections = page.sections

# dictionary with subsection
all_sections = page.table_of_contents

print(parse_links)


# all above is playing with api
