from datascience.config.configuration import *
from datascience.components.data_transformation import *
from datascience import logger
from pathlib import Path



STAGE_NAME = "Data Transformation Pipeline"


class DataTransformation:
    def __init__(self):
        pass

    def initiate_data_transformation(self):

        try:
            with open(Path("/Users/benjaminbrooke/PycharmProjects/MLOps/Big_project/artifacts/data_validation/status.txt"), "r") as f:
                status = f.read().split(" ")[-1]

            if status == "True":
                config = ConfigurationManager()
                data_transformation_config = config.get_data_transformation()
                data_transformation = DataTransformation(config=data_transformation_config)
                data_transformation.train_test_split()

            else:
                raise Exception("Your data scheme is valid")

        except Exception as e:
            print(e)
