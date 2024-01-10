import numpy as np
import os
import csv
from PIL import Image
import torch
import torchvision.transforms as transforms
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import random
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import LabelEncoder
import time
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import pickle
class CustomDataset():
    def __init__(self,data_list,transform):
        self.data_list = data_list
        self.label_encoder = LabelEncoder()
        self.label_binarizer = LabelBinarizer()
        self.transform = transform
        #Binarize Labels
        self.labels = self.label_binarizer.fit_transform([row[1] for row in self.data_list])
        # self.train_set,temp_data = train_test_split(self.data_list,train_size=0.75,test_size=0.25)
        # self.val_set,self.test_set = train_test_split(temp_data,test_size=0.05/(0.20+0.05))
        labels = [label for _, label in self.data_list]
        self.label_encoder.fit(labels)
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self,idx):
        img_path,labels = self.data_list[idx]
        img = Image.open(img_path).convert('RGB')
        if(self.transform):
            img = self.transform(img)
        # get binarized labels
        labels = self.labels[idx]
        # label_encoded = torch.tensor(self.label_encoder.transform([labels])[0])

        return img,labels

# Function to load the data
def load_data(data_dir):
    data_list = []

    for className in os.listdir(data_dir):
        class_path = os.path.join(data_dir,className)
        if (os.path.isdir(class_path)):
            for img_file in os.listdir(class_path):
                img_path = os.path.join(class_path,img_file)
                label = className
                data_list.append((img_path,label))
    return data_list

# Train the Model
def train_epoch(model, trainloader, criterion, device, optimizer):
    model.train()  # IMPORTANT!!!

    avg_loss = 0
    data_size = 0
    for batch_idx, data in enumerate(trainloader):

        inputs = data[0].to(device)
        labels = data[1].type(torch.float).to(device)
        outputs = model(inputs).to(device)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()  # reset accumulated gradients
        loss.backward()  # compute new gradients
        optimizer.step()  # apply new gradients to change model parameters

        avg_loss = (avg_loss * data_size + loss) / (data_size + inputs.shape[0])
        data_size += inputs.shape[0]
    return avg_loss

def train_modelcv(dataloader_cvtrain, dataloader_cvval, model, criterion, optimizer, scheduler, num_epochs, device):
    time_start = time.perf_counter()
    best_measure = 0
    best_epoch = -1
    val_losses = []
    train_losses = []
    val_acc = []
    all_class_acc = []
    all_class_prec_acc = []
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        losses = train_epoch(model, dataloader_cvtrain, criterion, device, optimizer)
        # scheduler.step()
        measure, val_loss,class_acc,all_prec_acc = evaluate(model, dataloader_cvval, criterion=criterion, device=device)
        print(f'Epoch {epoch+1} acc: ', measure.item())
        print(f'Epoch {epoch+1} train loss: ',losses.item())
        print(f'Epoch {epoch+1} val/test loss: ', val_loss.item())

        if measure > best_measure:  # if current acc > best recorded acc
            bestweights = model.state_dict()
            best_measure = measure
            best_epoch = epoch
            print('current best', measure.item(), ' at epoch ', best_epoch+1)
        all_class_acc.append(class_acc)
        all_class_prec_acc.append(all_prec_acc)
        losses = losses.detach().cpu().numpy()
        measure = measure.cpu().numpy()
        val_loss = val_loss.cpu().numpy()

        train_losses.append(losses.item())
        val_losses.append(val_loss)
        val_acc.append(measure.item())

    time_elapsed = time.perf_counter() - time_start
    return best_epoch, best_measure, bestweights,train_losses,val_losses,val_acc,all_class_acc,all_class_prec_acc,time_elapsed
def evaluate(model, dataloader, criterion, device):
    datasize = 0
    accuracy = 0
    avgloss = 0
    model.eval()  # IMPORTANT!!!
    y_true_ind = []
    y_scores = []
    y_proba = []
    with torch.no_grad():  # do not record computations for computing the gradient
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.type(torch.float)
            labels = labels.to(device)

            # get predictions
            outputs = model(inputs).to(device)
            if criterion is not None:
                curloss = criterion(outputs, labels)
                avgloss = (avgloss * datasize + curloss) / (datasize + inputs.shape[0])

            # get probabilities
            softmax_pred = torch.log_softmax(outputs,dim=1)
            # _,preds = torch.max(softmax_pred,dim=1)

            # get indice of predicted label which has highest probability
            preds = torch.argmax(softmax_pred,dim=1).data

            # get indice of label which has 1
            labels_idx = torch.argmax(labels,dim=1)
            # get 1 hot labels of preds
            one_hot_labels = torch.nn.functional.one_hot(preds, 10)

            accuracy = (accuracy * datasize + torch.sum(preds == labels_idx)) / (datasize + inputs.shape[0])
            datasize += inputs.shape[0]  # update datasize used in accuracy comp

            one_hot_labels = one_hot_labels.cpu().numpy()
            labels = labels.cpu().numpy()

            y_true_ind.append(labels)
            y_scores.append(one_hot_labels)
            y_proba.append(softmax_pred.cpu().numpy())
    if criterion is None:
        avgloss = None

    # For Precision calc
    all_class_prec = []
    for class_idx in range(10):
        class_prec = []
        for i in range(len(y_proba)):
            for j in range(len(y_proba[i])):
                if(y_true_ind[i][j][class_idx]==1.0):
                    ap = average_precision_score(y_true_ind[i][j], y_proba[i][j])
                    class_prec.append(ap)
        all_class_prec.append(sum(class_prec) / len(class_prec) * 100)

    # For class acc calc
    class_acc = []
    for element_idx in range(10):
        class_correct = 0
        class_count = 0
        for i in range(len(y_scores)):
            for j in range(len(y_scores[i])):
                if(y_true_ind[i][j][element_idx]) == 1.0 :
                    class_count += 1
                if(y_scores[i][j][element_idx] == 1.0 and y_true_ind[i][j][element_idx] == 1.0):
                    class_correct += 1
        # append the class acc
        class_acc.append(class_correct/class_count * 100)

    # if(phase == "test"):
    #     for class_idx in range(len(class_acc)):
    #         print(f"Class {class_idx} acc: {np.round(class_acc[class_idx],2)}%")

    return accuracy, avgloss,class_acc,all_class_prec

# Trg transform 1 Random Crop
trg_transforms1 = (transforms.Compose
([
    transforms.Resize(64),
    transforms.RandomCrop(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]))

# Val_ transform 1 Center Crop
val_transforms1 = (transforms.Compose
([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]))

# Train transform 2 Random Crop, Random Flip, Random Rotation, ColorJitter
trg_transforms2 = (transforms.Compose
([
    transforms.Resize(64),
    transforms.RandomCrop(64),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(45),
    transforms.ColorJitter(0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]))

# Val_transform 2 Random flip and random crop
val_transforms2 = (transforms.Compose
([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.RandomRotation(45),
    transforms.ColorJitter(0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]))

# Train transform 3 auto contrast
trg_transforms3 = (transforms.Compose
([
    transforms.Resize(64),
    transforms.RandomCrop(64),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(0.2),
    transforms.RandomAutocontrast(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]))
# Val_transform 3 Random flip and random crop
val_transforms3 = (transforms.Compose
([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]))
transform_list =[]
transform_list.append((trg_transforms1,val_transforms1))
transform_list.append((trg_transforms2,val_transforms2))
transform_list.append((trg_transforms3,val_transforms3))

def plot_train_val_graph(train_loss,val_loss,val_acc,num_epochs,filename):
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, num_epochs+1), train_loss, label=f' Training loss')
    plt.plot(range(1, num_epochs+1), val_loss, label=f'Val loss')
    # plt.plot(range(1, num_epochs+1), val_acc, label=f'Val acc')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title("Model Train-Val Loss")
    plt.legend()
    plt.savefig(filename)

if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    path = "..\\eurosat_dataset\\EuroSAT_RGB"
    mapped_data = load_data(path)
    # Set random seed to 5
    seed = 42
    gen = torch.Generator()
    gen.manual_seed(seed)
    train_set,val_set,test_set = torch.utils.data.random_split(mapped_data,[0.7,0.15,0.15],generator=gen)
    # train_set,temp_data = train_test_split(mapped_data,train_size=0.70,test_size=0.3,random_state=seed)
    # val_set,test_set = train_test_split(temp_data,test_size=0.5,random_state=seed)
    num_cl = 10
    num_epoch = 15
    test_model_list = []
    for transform_idx, transformations in enumerate(transform_list):
        train_dataset = CustomDataset(train_set,transform = transformations[0])
        val_dataset = CustomDataset(val_set, transform=transformations[1])
        dataloaders = \
        {
            'train' : torch.utils.data.DataLoader(train_dataset, batch_size=32,shuffle=True),
            'val': torch.utils.data.DataLoader(val_dataset, batch_size=32),
            # 'test': torch.utils.data.DataLoader(test_dataset, batch_size=32,shuffle=True)
        }
        print(f"##################### NEW RUN ############################")
        print(f'Using {device} for inference')
        print(f'-------------------------- Transform {transform_idx+1}: --------------------------------------')
        lr_list = [0.01,0.1]
        best_model_lr = []
        for lr in lr_list:
            model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
            # Modify last layer such that it outputs 10 classes
            model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 10)
            model.to(device)
            print(f"------------------ Current learning Rate: {lr}-------------------------------------------")
            # set loss criterion
            loss_crit = torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100,
                                                  reduce=None, reduction='mean')
            best_hyperparameter = None
            weights_chosen = None
            bestmeasure = None
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

            # get the train and val_loss
            (best_epoch, best_perfmeasure, bestweights, train_losses,
             val_losses,val_acc,val_class_acc,val_class_prec_acc,time_elapsed) = \
                (train_modelcv(dataloader_cvtrain=dataloaders['train'],
                      dataloader_cvval=dataloaders['val'],
                      model= model, criterion=loss_crit,
                      optimizer=optimizer, scheduler=None,
                      num_epochs=num_epoch, device=device))
            if best_hyperparameter is None:
                best_hyperparameter = lr
                weights_chosen = bestweights
                bestmeasure = best_perfmeasure
            elif best_perfmeasure > bestmeasure:
                best_hyperparameter = lr
                weights_chosen = bestweights
                bestmeasure = best_perfmeasure

            # end of for loop over hyperparameters here!
            # best model chosen
            model.load_state_dict(weights_chosen)
            plot_train_val_graph(train_losses,val_losses,val_acc,num_epoch,
                                 f"graph\\transform{transform_idx+1}_{lr}_trainval_loss.png")
            print(f"Best Model trained at Epoch {best_epoch+1} with an accuracy of {best_perfmeasure.item()}")
            print(f'Time elapsed for training and determining best model: {np.round(time_elapsed,0)}s')
            print("-" * 10)
            total_class_acc = []
            total_prec_acc = []
            for i in range(num_cl):
                class_acc = 0.0
                val_prec = 0.0
                # Get all class and val accuracies of the best epoch
                class_acc = val_class_acc[best_epoch][i]
                val_prec = val_class_prec_acc[best_epoch][i]
                total_class_acc.append(class_acc)
                total_prec_acc.append(val_prec)
            val_table = PrettyTable()
            val_table.field_names = ["Class","Val Set Class Accuracy","Val Set Class Precision"]
            for class_idx in range(len(total_class_acc)):
                val_table.add_row([f"{class_idx}",np.round(total_class_acc[class_idx],2),np.round(total_prec_acc[class_idx],2)])
                # print(f"Val set Class {class_idx} acc: {np.round(total_class_acc[class_idx], 2)}%")
                # print(f"Val set Class {class_idx} precision: {np.round(total_prec_acc[class_idx], 2)}%")
            print(val_table)

            print(f'Val set Classes Average acc: {np.sum(total_class_acc)/len(total_class_acc)}')
            print(f'Val set Classes Average precision: {np.sum(total_prec_acc) / len(total_prec_acc)}')
            print("-" * 10)
            # append the best model of each lr into a list
            best_model_lr.append((model.state_dict(), best_perfmeasure.item(),lr))
            # compare the best models whenever there are 2 in the list, pop out the lower accuracy one and
            # overwrite the saved model.
            if(len(best_model_lr) == 2):
                print(best_model_lr[0][1])
                print(best_model_lr[1][1])
                if(best_model_lr[0][1] < best_model_lr[1][1]):
                    print(f"Best Model saved acc: {best_model_lr[1][1]} at LR: {best_model_lr[1][2]}")
                    torch.save(best_model_lr[1][0],f'model\\best_model_transform{transform_idx+1}.pth')
                    best_model_lr.pop(0)
                else:
                    print(f"Best Model saved acc: {best_model_lr[0][1]} at LR: {best_model_lr[0][2]}")
                    torch.save(best_model_lr[0][0],f'model\\best_model_transform{transform_idx+1}.pth')
                    best_model_lr.pop(1)
        # append the best models of each transform into another list
        test_model_list.append((best_model_lr[0][0],best_model_lr[0][1],transform_idx))
        # Selects best model with best set for testing
        if(len(test_model_list)) == 2:
            if (test_model_list[0][1] < test_model_list[1][1]):
                test_model_list.pop(0)
            else:
                test_model_list.pop(1)
    # Save the best model,accuracy and transform idx for test
    with open('best_model.txt','wb') as fp:
        pickle.dump(test_model_list,fp)





