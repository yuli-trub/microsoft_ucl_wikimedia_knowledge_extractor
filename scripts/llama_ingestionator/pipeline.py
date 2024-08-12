from scripts.helper import log_duration
from llama_index.core.ingestion import IngestionPipeline, IngestionCache
from scripts.llama_ingestionator.transformator import (
    TextCleaner,
    SemanticChunkingTransformation,
    EntityExtractorTransformation,
    SummaryTransformation,
    KeyTakeawaysTransformation,
    EmbeddingTransformation,
    ImageDescriptionTransformation,
    PlotInsightsTransformation,
    ImageEntitiesTransformation,
)
import logging
from llama_index.core import Settings


# TODO: config cache
# TODO: add image processing later


def create_pipeline():

    # initialise the transformations
    text_cleaner = TextCleaner()
    semantic_chunking = SemanticChunkingTransformation()
    entities_extractor = EntityExtractorTransformation()
    summarisor = SummaryTransformation()
    key_takeaways = KeyTakeawaysTransformation()
    image_description = ImageDescriptionTransformation()
    plot_insights = PlotInsightsTransformation()
    image_entities = ImageEntitiesTransformation()
    embedding = EmbeddingTransformation()

    # ingestion pipeline
    pipeline = IngestionPipeline(
        transformations=[
            entities_extractor,
            summarisor,
            key_takeaways,
            image_description,
            plot_insights,
            image_entities,
            semantic_chunking,
            text_cleaner,
            embedding,
        ],
    )

    return pipeline


@log_duration
def run_pipeline(documents, pipeline, embed_model=Settings.embed_model):
    return pipeline.run(documents=documents, text_embed_model=embed_model)
