import logging
from dotenv import load_dotenv
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core.schema import TextNode
import os
from llama_index.core.node_parser import SemanticSplitterNodeParser


# Load environment variables
load_dotenv()
AZURE_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ENDPOINT = os.getenv("OPENAI_ENDPOINT")
ada_deployment_name = "text-embedding-ada-002"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    filename="test_pipeline.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Setup OpenAI embedding model
embed_model = AzureOpenAIEmbedding(
    model="text-embedding-ada-002",
    deployment_name="text-embedding-ada-002",
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=OPENAI_ENDPOINT,
    api_version="2023-07-01-preview",
)
# Initialize the SemanticChunker
splitter = SemanticSplitterNodeParser(
    buffer_size=1, breakpoint_percentile_threshold=80, embed_model=embed_model
)


sample_text = """Python was invented in the late 1980s by Guido van Rossum at Centrum Wiskunde & Informatica (CWI) in the Netherlands as a successor to the ABC programming language, which was inspired by SETL, capable of exception handling and interfacing with the Amoeba operating system. Its implementation began in December 1989. Van Rossum shouldered sole responsibility for the project, as the lead developer, until 12 July 2018, when 
he announced his "permanent vacation" from his responsibilities as Python's "benevolent dictator for life" (BDFL), a title the Python community bestowed upon him to reflect his long-term commitment as the project's chief decision-maker (he's since come out of retirement and is self-titled "BDFL-emeritus"). In January 2019, active Python core developers elected a five-member Steering Council to lead the project.
Python 2.0 was released on 16 October 2000, with many major new features such as list comprehensions, cycle-detecting garbage collection, reference counting, and Unicode support. Python 3.0, released on 3 December 2008, with many of its major features backported to Python 2.6.x and 2.7.x. Releases of Python 3 include the 2to3 utility, which automates the translation of Python 2 code to Python 3.
Python 2.7's end-of-life was initially set for 2015, then postponed to 2020 out of concern that a large body of existing code could not easily be forward-ported to Python 3. No further security patches or other improvements will be released for it. Currently only 3.8 and later are 
supported (2023 security issues were fixed in e.g. 3.7.17, the final 3.7.x release). While Python 2.7 and older is officially unsupported, a 
different unofficial Python implementation, PyPy, continues to support Python 2, i.e. "2.7.18+" (plus 3.9 and 3.10), with the plus meaning (at least some) "backported security updates".
In 2021 (and again twice in 2022), security updates were expedited, since all Python versions were insecure (including 2.7) because of security issues leading to possible remote code execution and web-cache poisoning. In 2022, Python 3.10.4 and 3.9.12 were expedited and 3.8.13, because of many security issues. When Python 3.9.13 was released in May 2022, it was announced that the 3.9 series (joining the older series 3.8 and 3.7) would only receive security fixes in the future. On 7 September 2022, four new releases were made due to a potential denial-of-service attack: 3.10.7, 3.9.14, 3.8.14, and 3.7.14.
As of October 2023, Python 3.12 is the stable release, and 3.12 and 3.11 are the only versions with active (as opposed to just security) support. Notable changes in 3.11 from 3.10 include increased program execution speed and improved error reporting.
Every Python release since 3.5 has added some syntax to the language. 3.10 added the | union type operator  and the match and case keywords (for structural pattern matching statements). 3.11 expanded exception handling functionality. Python 3.12 added the new keyword type.
Python 3.11 claims to be between 10 and 60% faster than Python 3.10, and Python 3.12 adds another 5% on top of that. It also has improved error messages, and many other changes.
Since 27 June 2023, Python 3.8 is the oldest supported version of Python (albeit in the 'security support' phase), due to Python 3.7 reaching end-of-life.
Python 3.13 introduced an incremental garbage collector (producing shorter pauses for collection in programs with a lot of objects); an experimental JIT compiler; and removals from the C API. Some standard library modules and many deprecated classes, functions and methods, will be 
removed in Python 3.15 and or 3.16. Starting with 3.13, it and later versions have 2 years of full support (up from one and a half); followed by 3 years of security support (for same total support as before)."""
sample_node = TextNode(
    text=sample_text,
    metadata={
        "id": "sample_section",
        "type": "section",
        "title": "Sample Section",
    },
)


# Process the sample node
chunks = splitter.get_nodes_from_documents([sample_node])

# Print the results
print("Original Node:")
print(f"ID: {sample_node.metadata['id']}")
print(f"Content: {sample_node.text}\n")

print("Processed Chunks:")
# print(f" Chunks: {chunks}")
for i, chunk in enumerate(chunks):

    print(f"Chunk {i+1}:")
    print(f"ID: {chunk.metadata['id']}")
    print(f"Parent ID: {chunk.metadata}")
    print(f"Content: {chunk.text}\n")
