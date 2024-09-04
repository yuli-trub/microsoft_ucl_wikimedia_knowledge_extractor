# Wikimedia Multimodal Knowledge Extraction and Retrieval System

## Overview

This project implements a multimodal knowledge extraction and retrieval system designed to pull data from Wikimedia sources, preprocess it, and store it in structured formats to enable efficient querying and retrieval. The system integrates LlamaIndex for indexing and retrieval, Neo4j for graph-based storage, and Qdrant for vector-based storage. It supports the processing of both text and images, allowing for complex multimodal queries and retrieval tasks.

The project is structured into two primary components:

- **Knowledge Extractor**: Handles data ingestion, preprocessing, transformation, and storage.
- **Retriever**: Retrieves the stored data based on user queries and delivers enhanced responses using external LLMs (OpenAI, Azure OpenAI).

## Features

- **Multimodal Data Integration**: The system successfully integrates both textual and visual data, enhancing the relevance and accuracy of AI-generated responses. This capability is crucial for handling complex queries that require an understanding of multiple data formats.

- **Hybrid Storage Solution**: The system uses a combination of Neo4j for graph-based data and Qdrant for vector embeddings. This dual approach allows for efficient retrieval of structured and unstructured data, ensuring that the system can deliver context-rich, detailed responses.

- **Custom Retrieval Mechanisms**: The project implements a fusion retrieval mechanism, which generates multiple queries to cover diverse aspects of the user input. A custom reranking process ensures the most relevant results are prioritised.

- **ReAct Agent Integration**: By incorporating the ReAct agent, the system ensures dynamic adjustments based on real-time context. This allows for more consistent, predictable, and contextually appropriate responses.

- **Containerisation with Docker**: The entire system is containerised using Docker, with Docker Compose managing the orchestration of the different services. This ensures consistent deployment across different environments, enhancing the system’s maintainability and scalability.


## Tech Stack

The project uses a modern and scalable stack to ensure reliability and performance:

- **[LlamaIndex](https://llamaindex.ai/docs)**: For creating an index of the processed data and performing efficient queries.
- **[Neo4j](https://neo4j.com/docs/)**: A graph database to store relationships between different nodes, ensuring flexible and intuitive data retrieval.
- **[Qdrant](https://qdrant.tech/documentation/)**: A vector database to store embeddings for text and images, enabling fast and accurate similarity searches.
- **[OpenAI / Azure OpenAI](https://learn.microsoft.com/azure/ai-services/openai/overview)**: For generating enhanced responses using language models such as GPT-4O.
- **[Docker](https://docs.docker.com/get-started/)**: For containerizing the application, making it easy to deploy and manage in different environments.
- **[Docker Compose](https://docs.docker.com/compose/)**: To orchestrate the various services (knowledge extractor, retriever, Neo4j, Qdrant) and manage them as a cohesive system.
- **[Python](https://docs.python.org/3/)**: Core language for data processing, transformation, and integration with external services.

## System Architecture


![System Architecture](/img/system_diagram.png)


The system consists of the following core components:

- **Data Ingestion**: Gathers data from Wikimedia sources such as text, images, and tables.
- **Image Classification and Transformation Pipeline**: Processes and classifies images before converting them into structured nodes using LLamaindex and models deployed on Azure OpenAI resource
- **Storage Management**: Uses Neo4j for graph-based storage and Qdrant for vector-based storage of embeddings. This ensures quick retrieval of knowledge nodes.
- **Retrieval System**: A query engine that retrieves relevant multimodal data (text and images) and generates enhanced responses using language models.


## Installation

### Prerequisites

- **Azure Account**: If you're new to Azure, you can [get an Azure account for free](https://aka.ms/free) and receive some free Azure credits to get started.
- **Azure Subscription**: Ensure you have an Azure subscription with access enabled for the Azure OpenAI Service. For more details, see the [Azure OpenAI Service documentation](https://learn.microsoft.com/azure/ai-services/openai/overview#how-do-i-get-access-to-azure-openai) on how to get access.
- **Azure OpenAI Resource**: For this project, you'll need to deploy GPT-4 and text-embedding-ada-002 models with Azure OpenAI Resource and Computer Vision Resource. Refer to the [Azure OpenAI Service documentation](https://learn.microsoft.com/azure/ai-services/openai/how-to/create-resource?pivots=web-portal) for more details on deploying models and [model availability](https://learn.microsoft.com/azure/ai-services/openai/concepts/models).
- **Docker**: Make sure [Docker](https://docs.docker.com/get-docker/) is installed on your machine.
- **Docker Compose**: Required for managing the multi-container setup. You can install it by following the instructions [here](https://docs.docker.com/compose/install/).

### Environment Setup

Clone this repository:

```bash
git clone https://github.com/your-repo/knowledge-extraction.git
cd knowledge-extraction 
```

Create a `.env` file for both the `knowledge_extractor`and  `retriever` containers and root directory. Refer to the provided `.env.example` files in the respective folders.


## Docker Configuration

### Knowledge Extractor

The `knowledge_extractor` container handles all tasks related to data ingestion, image classification, transformation, and storage. It interacts with the Neo4j and Qdrant databases for storing graph and vector-based data.

### Retriever

The `retriever` container processes user queries, retrieves relevant nodes, and delivers enhanced responses. It interacts with external APIs like OpenAI and Azure OpenAI for generating language model responses.

### Running the system

Build the Docker containers:

```bash
docker-compose build
```

Start the containers:

```bash
docker-compose up
```

Access the Gradio interface at `http://localhost:<PORT>` to interact with the retrieval system.



## Usage

Once the containers are up and running, you can interact with the retriever service via a Gradio interface. This interface allows users to input questions, and the system will generate both enhanced and standard responses based on the stored knowledge.

### Example Query

Navigate to `http://localhost:<PORT>` as specified in the `.env` file (default is port 5000).  
Type a question in the provided Gradio textbox. For example for domain topic about Eiffel Tower:

```text
"When was the Eiffel Tower built, and what was its original purpose?" 
```

The system will generate:

- **Enhanced Response**: Incorporates retrieved multimodal data (text, images, tables).
- **Standard Response**: Generated using a language model without retrieved context.

![Gradio UI](/img/gradio.png)

## Directory Structure

```bash
knowledge_extractor/
├── app_logs/               # Logs for system execution
├── data/                   # Input data files
├── logs/                   # General logs for debugging
├── scripts/                # Core knowledge extraction scripts
├── Dockerfile              # Dockerfile for knowledge extractor container
├── main.py                 # Entry point for knowledge extractor container
├── .env                    # Your environment variables
├── .env.example            # Env example
└── requirements.txt        # Python dependencies for the knowledge extractor

retriever/
├── agents/                 # Agents for query processing
├── app_logs/               # Logs for the retriever system
├── flagged/                # Logs or flagged data 
├── scripts/                # Core retriever scripts
├── Dockerfile              # Dockerfile for retriever container
├── main.py                 # Entry point for retriever container
├── .env                    # Your environment variables
├── .env.example            # Env example
└── requirements.txt        # Python dependencies for the retriever

docker-compose.yml          # Docker Compose configuration for managing both containers
.env                        # Your environment variables
.env.example                # Env example
```

## Future Work

- **Agentic Framework Expansion**: Future iterations could involve a tool that autonomously determines if the existing knowledge base is sufficient to answer a query. If not, the system would initiate the knowledge extraction process to create new data on the topic, making the system more adaptive and self-sufficient.

- **Knowledge Graph with Entities**: A more detailed knowledge graph could be developed, incorporating key entities like people, places, and events. This would improve the depth and accuracy of retrieval processes by creating richer, more interconnected knowledge representations.

- **Enhanced Multimodal Capabilities**: Further development could enable functionalities like image comparison, object detection, and splitting images into separate objects, enhancing the system's ability to handle a wider variety of data formats.
