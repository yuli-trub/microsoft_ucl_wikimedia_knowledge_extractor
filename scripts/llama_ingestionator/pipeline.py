from helper import log_duration
from llama_index.core.ingestion import IngestionPipeline, IngestionCache
from llama_ingestionator.transformator import (
    TextCleaner,
    SemanticChunkingTransformation,
    EntityExtractorTransformation,
    SummaryTransformation,
    KeyTakeawaysTransformation,
    EmbeddingTransformation,
)
import logging
from llama_index.core import Settings


# TODO: config cache
# TODO: add image processing later


def create_pipeline(vector_store):

    # initialise the transformations
    text_cleaner = TextCleaner()
    semantic_chunking = SemanticChunkingTransformation()
    entities_extractor = EntityExtractorTransformation()
    summarisor = SummaryTransformation()
    key_takeaways = KeyTakeawaysTransformation()
    embedding = EmbeddingTransformation()

    # ingestion pipeline
    pipeline = IngestionPipeline(
        transformations=[
            text_cleaner,
            entities_extractor,
            summarisor,
            key_takeaways,
            semantic_chunking,
            embedding,
        ],
        # cache=ingest_cache,
        vector_store=vector_store,
    )

    return pipeline


@log_duration
def run_pipeline(documents, pipeline, embed_model=Settings.embed_model):
    return pipeline.run(documents=documents, text_embed_model=embed_model)
