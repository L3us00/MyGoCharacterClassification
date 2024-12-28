import matplotlib.pyplot as plt
import torch
from src.resnet.constants import TRAIN_CONFIG, DEVICE, DATACLASS
from src.resnet.resnet_model.classification_model import ClassificationModel
from src.resnet.utils.get_dataloader import get_dataloader
import numpy as np

def plot_classification_results(model, data_loader, device, num_images=9):

    # Set up a 3x3 grid plot
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()

    # Get a batch of images and labels
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(data_loader):
            if batch_idx == 1:  # Only get one batch
                break
            print(labels)
            images, labels = images.to(device), torch.tensor([int(label) for label in labels], dtype=torch.long, device=DEVICE)


            # Make predictions
            preds = model.predict(images)

            # Loop through the images in the batch and plot them
            for i in range(min(num_images, len(images))):
                ax = axes[i]
                img = images[i].cpu().permute(1, 2, 0).numpy()
                img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                img = np.clip(img, 0, 1)
                ax.imshow(img)
                correct = preds[i].argmax(dim=0) == labels[i]
                color = 'green' if correct else 'red'
                ax.set_title(f"Pred: {DATACLASS[preds[i].argmax(dim=0).item()]}", color=color)
                ax.axis('off')

    plt.savefig(r'assets/valid_result2.png', bbox_inches='tight') 

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()  # Only needed if you're freezing the script into an executable


    train_loader, val_loader = get_dataloader()
    model_resnet = ClassificationModel(TRAIN_CONFIG["classification_class"],False)
    model_resnet.load(TRAIN_CONFIG["model_path"])

    plot_classification_results(model_resnet,data_loader=val_loader,device=DEVICE)
