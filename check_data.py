from dotenv import load_dotenv
load_dotenv()

import os

DATA_DIR = 'data'
DATASET_FILE = 'salaries.csv'
file_path = os.path.join(DATA_DIR, DATASET_FILE)

if os.path.isfile(file_path):
	print(f"Dataset found: {file_path}")
else:
	print(f"Dataset NOT found in '{DATA_DIR}'. Please download '{DATASET_FILE}' and place it in the '{DATA_DIR}' folder.")