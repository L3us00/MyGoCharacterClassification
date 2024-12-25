import torch


def accuracy_fn(y_pred,y_true):
    # Get the predicted class by finding the index with the maximum value
    y_pred_class = torch.argmax(y_pred)
    
    # Check how many predictions are correct
    correct_predictions = torch.eq(y_pred_class, y_true).sum().item()
    
    # Calculate the accuracy
    accuracy = (correct_predictions / len(y_true)) * 100
    
    return accuracy