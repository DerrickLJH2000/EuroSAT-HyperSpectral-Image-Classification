import numpy as np
import torch
from validate import validate

# Define Training Loop
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
   
    counter = 0
    train_running_loss = 0.0

    for batch_idx, data in enumerate(dataloader):
        counter += 1
        inputs, labels = data[0].to(device), data[1].type(torch.float).to(device)
        optimizer.zero_grad()

        outputs = model(inputs)

        # apply sigmoid activation to get all the outputs between 0 and 1
        # outputs = torch.sigmoid(outputs)
        loss = criterion(outputs, labels)

        train_running_loss += loss.item()
        # back propagation
        loss.backward()

        # update optimizer parameters
        optimizer.step()

    train_loss = train_running_loss / counter

    return train_loss


def train_cv_model(model, dataloader_train, dataloader_val, criterion, optimizer, device, num_epochs):
    train_losses = []
    val_losses = []
    val_acc = []

    bestweights = None
    best_measure = 0
    best_epoch = -1
    
    for epoch in range(num_epochs):
        print('\nEpoch [{}/{}]'.format(epoch+1, num_epochs))
        print('-' * 10)     

        train_epoch_loss = train_epoch(
            model, dataloader_train, criterion, optimizer, device
        )

        measure, valid_epoch_loss, all_class_precision, class_accuracy, hamming_loss = validate(
            model, dataloader_val, criterion, device
        )

        train_losses.append(train_epoch_loss)
        val_losses.append(valid_epoch_loss)
        mAP = np.mean(all_class_precision)
        mAcc = np.mean(class_accuracy)

        print(f'Subset Accuracy: {measure:.4f}')
        print(f'Hamming Loss: {hamming_loss:.4f}')
        print(f'Training Loss: {train_epoch_loss:.4f}')
        print(f'Validation Loss: {valid_epoch_loss:.4f}')
        print(f'\nAccuracy (per class):')
        for i, acc in enumerate(class_accuracy):
            print(f'Class {i}: {acc:.4f}')

        print(f'Average Precision Score (per class):')
        for i, ap in enumerate(all_class_precision):
            print(f'Class {i}: {ap:.4f}')

        print(f'Mean AP Score Over All Classes: {mAP}')       
        print(f'Mean Accuracy Over All Classes: {mAcc}')       


        
        # store current parameters because they are the best or not?
        if measure > best_measure:
            bestweights= model.state_dict()
            best_measure = measure
            best_epoch = epoch

        val_acc.append(measure)

    return best_epoch, best_measure, bestweights, train_losses, val_losses
