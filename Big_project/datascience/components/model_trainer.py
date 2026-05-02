import pandas as pd
import os
from datascience import logger
from sklearn.linear_model import ElasticNet
import joblib
from datascience.entity.config_entity import *



class ModelTrainer:
    def __init__(self,config: DataModelConfig):
        self.config = config

    def train(self):
        train_data = pd.read_csv(self.config.train_path,sep=";")
        test_data = pd.read_csv(self.config.test_path,sep=";")

        print(train_data.columns)
        print(test_data.columns)

        print("passed - first")
        print(self.config.target_column)
        train_x = train_data.drop(self.config.target_column, axis=1)
        test_x = test_data.drop(self.config.target_column, axis=1)
        print("passed - third")
        train_y = train_data[[self.config.target_column]]
        test_y = test_data[[self.config.target_column]]
        print("passed - second")

        lr = ElasticNet(alpha=self.config.alpha, l1_ratio=self.config.l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        joblib.dump(lr, os.path.join(self.config.root_dir, self.config.model_name))
