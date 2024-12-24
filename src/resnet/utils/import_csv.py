import csv
from typing import List, Dict, Tuple
from ..constants import DATASET_LABEL_PATH
from sklearn.model_selection import train_test_split


def import_csv() -> List[Dict]:
    with open(DATASET_LABEL_PATH, mode='r') as file:
        csv_reader = csv.DictReader(file)
        data = [row for row in csv_reader]
    return data

def split_data(
        data: List[Dict], 
        test_size: float = 0.2, 
        random_state: int = 42
        ) -> Tuple[List[Dict], List[Dict]]:
    train_data, val_data = train_test_split(data, test_size=test_size, random_state=random_state)
    return train_data, val_data