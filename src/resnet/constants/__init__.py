import os 
from ..utils.load_config import load_config
import torch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_DIR = os.path.join("assets", "dataset")
DATACLASS = {
    0:"anon",
    1:"tomori",
    2:"taki",
    3:"soyo",
    4:"rana",
    5:"saki",
    6:"muzimi",
    7:"umirin",
    8:"ueka",
    9:"nyamuchi",
    10:"other"
}
DATASET_LABEL_PATH = os.path.join("assets", "dataset", "label.csv")
CONFIG_PATH = os.path.join("config", "train_config.yaml")
TRAIN_CONFIG = load_config(CONFIG_PATH)
EPSILON = TRAIN_CONFIG.get("eps", 1e-5)
MOMENTUM = TRAIN_CONFIG.get("momentum", 0.1)
LR = TRAIN_CONFIG.get("lr", 0.01)
