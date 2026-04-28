import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

project_name = "datascience"

list_of_files = [".github/workflows/.gitkeep",
                 f"{project_name}/__init__.py",
                 f"{project_name}/coponents/__init__.py",
                 f"{project_name}/utils/__init__.py",
                 f"{project_name}/utils/common.py",
                 f"{project_name}/config/__init__.py",
                 f"{project_name}/config/configuration.py",
                 f"{project_name}/pipeline/__init__.py",
                 f"{project_name}/entity/__init__.py",
                 f"{project_name}/entity/config_entity.py",
                 f"{project_name}/constants/__init__.py",
                 "config/config.yaml",
                 "params.yaml",
                 "schema.yaml",
                 "main.py",
                 "DockerFile",
                 "research/research.ipynb",
                 "templates/inde.html"
                 ]

for filepath in list_of_files:
    filepath = Path(f"/Users/benjaminbrooke/PycharmProjects/MLOps/Big_project/{filepath}")
    filedir,filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir,exist_ok = True)
        logging.info(f"Created {filedir} for the file {filename}")

    if (not os.path.exists(filepath)) or (not os.path.isfile(filepath) == 0):
        with open(filepath,"w") as f:

            pass

            logging.info(f"Created empty file: {filepath}")

    else:
        logging.info(f"Skipping {filepath}")




