import logging


def setup_logging():
    loggers = {
        "transformator": "transformator.log",
        "ingestionator": "ingestionator.log",
        "documentifier": "documentifier.log",
        "storage_manager": "storage_manager.log",
        "graph_db": "graph_db.log",
        "pipeline_logger": "pipeline.log",
    }

    logging.getLogger().handlers = []

    for logger_name, log_file in loggers.items():
        logger = logging.getLogger(logger_name)

        if not logger.hasHandlers():
            handler = logging.FileHandler(log_file)
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)


setup_logging()
