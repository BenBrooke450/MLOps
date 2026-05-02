from datascience.config.configuration import *
from datascience.components.data_transformation import *
from datascience import logger
from pathlib import Path
from datascience.components.model_trainer import *




STAGE_NAME = "Model Trainer Stage"



class ModelTrainerPipeline():
    def __init__(self):
        pass

    def initiate_model_trainer(self):
        config = ConfigurationManager()
        data_trainer_config = config.get_model_training()
        data_trainer = ModelTrainer(config=data_trainer_config)
        data_trainer.train()


