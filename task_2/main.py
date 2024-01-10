import numpy as np
import os
import torch
import torchvision.transforms as transforms
import random
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights
import matplotlib.pyplot as plt
from dataset import CustomDataset, load_data
from torch.utils.data import DataLoader, random_split
from train import train_cv_model
from validate import validate

# Plot train and validation losses
def plot_train_val_graph(train_loss,val_loss,num_epochs, filename):
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, num_epochs+1), train_loss, label=f' Training Loss')
    plt.plot(range(1, num_epochs+1), val_loss, label=f'Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title("Model Train-Val Loss")
    plt.legend()
    plt.savefig(filename)


# Data Augmentation Schema
transform1 = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    
    # Modify the classifier for binary classification
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier[1].in_features, 10),
        nn.Sigmoid()
    )

    model = model.cuda().to(device)

    # Load data
    path = "..\\eurosat_dataset\\EuroSAT_RGB"
    mapped_data = load_data(path)

    seed = 42
    gen = torch.Generator()
    gen.manual_seed(seed)
    train_set,val_set,test_set = random_split(mapped_data,[0.7,0.15,0.15],generator=gen)

    train_dataset = CustomDataset(train_set, transform=transform1)
    val_dataset = CustomDataset(val_set, transform=transform1)
    test_dataset = CustomDataset(test_set, transform=transform1)

    print(f'Using {device} for inference')
    dataloaders = {
        'train' : DataLoader(train_dataset, batch_size=32, shuffle=True),
        'val': DataLoader(val_dataset, batch_size=32, shuffle=False),
        'test': DataLoader(test_dataset, batch_size=32, shuffle=False)
    }

    # Define hyperparameters
    num_epochs = 10
    lrates = [0.001, 0.01, 0.1]

    criterion = nn.BCELoss()


    best_hyperparameter = None
    weights_chosen = None
    best_measure = None
    best_epoch = None
    best_train_losses = None
    best_val_losses = None

    for lr in lrates:

        print(f'\n\n\n################### NEW RUN ##################')
        print(f'############ LEARNING RATE: {lr} ############')
        print(f'##############################################')

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        best_epoch, best_perfmeasure, bestweights, train_losses, val_losses = train_cv_model(
            model, dataloaders['train'], dataloaders['val'], criterion, optimizer, device, num_epochs)
        folder_path = 'training_outputs/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        graph_path = os.path.join(folder_path, f'train_val_loss_curve_{lr}.png')
        plot_train_val_graph(train_losses, val_losses, num_epochs, graph_path)

        if best_hyperparameter is None or best_perfmeasure > best_measure:
            best_hyperparameter = lr
            best_measure = best_perfmeasure
            best_weights = bestweights
            best_epoch = best_epoch
            best_train_losses = train_losses
            best_val_losses = val_losses
            print('\ncurrent best', best_measure, ' at epoch ', best_epoch)


    # Save the model with the best performance, hyperparameters, and optimizer
    folder_path = 'outputs/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    model_path = os.path.join(folder_path, 'best_model.pth')
    torch.save({
        'epoch': best_epoch,
        'model_state_dict': best_weights,
        'best_measure': best_measure,
        'best_hyperparameter': best_hyperparameter
    }, model_path)

    print(f"Saved best model at learning rate {best_hyperparameter} epoch {best_epoch+1} with accuracy {best_measure:.4f}")

    graph_path = os.path.join(folder_path, f'train_val_loss_curve_{best_hyperparameter}.png')
    # Plot the train and val loss curve for the best model
    plot_train_val_graph(best_train_losses, best_val_losses, num_epochs, graph_path)
    
    # Load the saved model
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Now, you can use the validate function for testing
    test_measure, test_loss, test_ap_per_class, test_class_accuracy, test_hamming_loss = validate(model, dataloaders['test'], criterion, device)

    test_mAP = np.mean(test_ap_per_class)
    test_mAcc = np.mean(test_class_accuracy)

    test_result_path = os.path.join(folder_path, 'test_results.txt')

    # Open a file for writing
    with open(test_result_path, 'w') as file:
        # Write the results to the file
        file.write(f'Best Model Subset Accuracy: {best_measure:.4f}')
        file.write(f'\nTest Subset Accuracy: {test_measure:.4f}')
        file.write(f'\nTest Hamming Loss: {test_hamming_loss:.4f}')
        file.write(f'\nTest Validation Loss: {test_loss:.4f}\n')

        file.write(f'\n\nTest Accuracy (per class)')
        for i, acc in enumerate(test_class_accuracy):
            file.write(f'\nClass {i}: {acc:.4f}')

        file.write(f'\n\nTest AP Score (per class)')
        for i, ap in enumerate(test_ap_per_class):
            file.write(f'\nClass {i}: {ap:.4f}')

        file.write(f'\n\nTest Mean AP Score Over All Classes: {test_mAP}')
        file.write(f'\nTest Mean Accuracy Over All Classes: {test_mAcc}')  
        print(f"Test Results written to {test_result_path}")

    file.close()
