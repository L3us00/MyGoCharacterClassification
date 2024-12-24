import csv
from typing import List, Dict
from ..constants import DATASET_LABEL_PATH


def to_csv(data: List[Dict]):
    with open(DATASET_LABEL_PATH, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["file_path", "class"])
        writer.writeheader()  # Write the header row
        writer.writerows(data)  # Write the data rows

    print(f"Data saved to {DATASET_LABEL_PATH}")