from datascience.constants import *
from datascience.utils.common import *
from datascience.entity.config_entity import *
import os
from datascience import logger
from sklearn.model_selection import train_test_split
import pandas as pd


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def train_test_split_func(self):
        data = pd.read_csv(self.config.source,sep=";")

        train, test = train_test_split(data)

        train.to_csv(os.path.join(self.config.root_dir, "train.csv"),sep=";", index=False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"),sep=";", index=False)

        logger.info("----Split data-----")
        logger.info(train.shape)
        logger.info(test.shape)

        print(train.shape)
        print(test.shape)