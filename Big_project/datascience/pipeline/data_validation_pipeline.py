from datascience.config.configuration import *
from datascience import logger
from datascience.components.data_validation import *

STAGE_NAME = "data_validation"

class DataValidationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_validation_config = config.get_data_validation()
        data_validation = DataValidation(config=data_validation_config)
        data_validation.validate_all_columns()



if __name__ == "__main__":
    try:
        logger.info(F"------- stage: {STAGE_NAME} STARTED-------")
        obj = DataValidationPipeline()
        obj.main()
        logger.info(F"------- stage: {STAGE_NAME} FINISHED-------")
    except Exception as e:
        logger.error(e)