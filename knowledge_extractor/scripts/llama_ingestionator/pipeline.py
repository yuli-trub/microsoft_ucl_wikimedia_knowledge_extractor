from scripts.helper import log_duration
from llama_index.core.ingestion import IngestionPipeline
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
    TableAnalysisTransformation
)
from llama_index.core import Settings



def create_pipeline():
    """Create the ingestion pipeline"""

    # initialise the transformations
    text_cleaner = TextCleaner()
    semantic_chunking = SemanticChunkingTransformation()
    entities_extractor = EntityExtractorTransformation()
    summarisor = SummaryTransformation()
    key_takeaways = KeyTakeawaysTransformation()
    image_description = ImageDescriptionTransformation()
    plot_insights = PlotInsightsTransformation()
    image_entities = ImageEntitiesTransformation()
    table_analysis = TableAnalysisTransformation()
    embedding = EmbeddingTransformation()

    # ingestion pipeline
    pipeline = IngestionPipeline(
        transformations=[
            entities_extractor,
            summarisor,
            key_takeaways,
            image_description,
            image_entities,
            plot_insights,
            table_analysis,
            semantic_chunking,
            text_cleaner,
            embedding,
        ],
    )

    return pipeline


@log_duration
def run_pipeline(documents, pipeline, embed_model=Settings.embed_model):
    """Run the ingestion pipeline on a list of documents"""
    return pipeline.run(documents=documents, text_embed_model=embed_model)
