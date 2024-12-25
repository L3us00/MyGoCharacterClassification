from src.resnet.resnet_model.classification_model import ClassificationModel
import torch.nn as nn
import torch
from ..utils.get_dataloader import get_dataloader
from ..utils.accuracy_fn import accuracy_fn
from ..constants import LR, TRAIN_CONFIG
from torchvision import models
from tqdm import tqdm


def train():
    torch.manual_seed(TRAIN_CONFIG.get("random_seed",42))
    train_loader, val_loader = get_dataloader()

    model_resnet = ClassificationModel()
    model_resnet.train(train_loader=train_loader,val_loader=val_loader)
