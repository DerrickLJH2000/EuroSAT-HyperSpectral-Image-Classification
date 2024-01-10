import numpy as np
import torch
from sklearn.metrics import average_precision_score, hamming_loss


def validate(model, dataloader, criterion, device):
    model.eval()
    threshold = 0.6

    counter = 0
    val_running_loss = 0.0

    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            counter += 1
            inputs, labels = inputs.to(device), labels.type(torch.float).to(device)
            outputs = model(inputs)
            if criterion is not None:
                # apply sigmoid activation to get all the outputs between 0 and 1
                # outputs = torch.sigmoid(outputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
            
            # Apply threshold to get predicted labels
            predicted = (torch.sigmoid(outputs) >= threshold).float()

            y_true.append(labels.cpu().numpy())
            y_pred.append(predicted.cpu().numpy())      

        y_true = np.concatenate(y_true, axis=0)
        y_pred = np.concatenate(y_pred, axis=0)

        subset_accuracy = np.mean(np.all(y_true == y_pred, axis=1)) * 100
        hamming_loss_value = hamming_loss(y_true, y_pred)

        # Calculate Average Precision
        all_class_precision = []
        for class_idx in range(10):
            class_y_true = y_true[:, class_idx]
            class_y_pred = y_pred[:, class_idx]
            if 1 in class_y_true:
                ap = average_precision_score(class_y_true, class_y_pred)
                all_class_precision.append(ap * 100)

        # Calculate Accuracy per class
        class_accuracy = []
        for class_idx in range(10):
            correct = 0
            total = 0
            class_y_true = y_true[:, class_idx]
            class_y_pred = y_pred[:, class_idx]
            correct += np.sum(class_y_true == class_y_pred)
            total = len(class_y_true)
            class_accuracy.append(correct / total * 100)

    val_loss = val_running_loss / counter

    return subset_accuracy, val_loss, all_class_precision, class_accuracy, hamming_loss_value
