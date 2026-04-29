
from datascience.config.configuration import *
from datascience.components.data_ingestion import *
from datascience import logger

STAGE_NAME = "Data Ingestion Pipeline"


class DataIngestionTrainingP:
    def __init__(self):
        pass

    def intiate_data_ingestion(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()


if __name__ == "__main__":
    try:
        logger.info(F"------- stage: {STAGE_NAME} STARTED-------")
        obj = DataIngestionTrainingP()
        obj.intiate_data_ingestion()
        logger.info(F"------- stage: {STAGE_NAME} FINISHED-------")
    except Exception as e:
        logger.error(e)