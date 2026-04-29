import requests
import os
import urllib.request as request
from datascience import logger
import zipfile
from datascience.entity.config_entity import *



class DataIngestion:
    def __init__(self,config: DataIngestionConfig):
        self.config = config

    def download_file(self):

        os.makedirs(os.path.dirname(self.config.local_data_file), exist_ok=True)

        if not os.path.exists(self.config.local_data_file):
            filename, headers = request.urlretrieve(
                url=self.config.source_url,
                filename= self.config.local_data_file)
            logger.info(f"{filename} download! with following info: \n{headers}")
        else:
            logger.info(f"File already exists")
            print(self.config.local_data_file)

