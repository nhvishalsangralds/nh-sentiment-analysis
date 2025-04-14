import os
from pathlib import Path
import logging


# Define project name
project_name = "predictor"

# List of required files and folders
list_of_items = [
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/preprocessing.py",
    f"src/{project_name}/pipelines/__init__.py",
    f"src/{project_name}/utils.py"  # Correctly handle folder creation
]

# Iterate through each path
for item in list_of_items:
    path = Path(item)

    # If item is a directory (ends with '/')
    if item.endswith("/"):
        path.mkdir(parents=True, exist_ok=True)
        logging.info(f"✅ Created directory: {path}")

    # If item is a file
    else:
        path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the parent directory exists
        if not path.exists():
            path.touch()  # Create an empty file
            logging.info(f"📄 Created file: {path}")
        else:
            logging.info(f"⚠️ File already exists: {path}")