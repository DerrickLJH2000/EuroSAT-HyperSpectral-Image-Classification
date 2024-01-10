import numpy as np
import os
import io
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

def save_data(data_list,pred_labels):
    with open('test_predictions.txt','w') as file:
        for idx in range(len(pred_labels)):
            line = f"{data_list[idx][0]} {pred_labels[idx]} {data_list[idx][1]}\n"
            file.write(line)
def load_data(data_dir):
    data_list = []
    className_list = []
    for className in os.listdir(data_dir):
        class_path = os.path.join(data_dir,className)
        if (os.path.isdir(class_path)):
            for img_file in os.listdir(class_path):
                img_path = os.path.join(class_path,img_file)
                label = className
                data_list.append((img_path,label))
        className_list.append(className)
    return data_list,className_list

# Custom CPU unpickler for CPU only machines
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)
def evaluate(model, dataloader, criterion, device):
    datasize = 0
    accuracy = 0
    avgloss = 0
    model.eval()  # IMPORTANT!!!
    y_true_ind = []
    y_scores = []
    y_proba = []
    all_predictions = []
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

            # get indice of predicted label which has highest probabilit
            preds = torch.argmax(softmax_pred,dim=1).data

            # get indice of label which has 1
            labels_idx = torch.argmax(labels,dim=1)

            # get 1 hot labels of preds
            one_hot_labels = torch.nn.functional.one_hot(preds, 10)

            for batch_length in range(len(one_hot_labels)):
                all_predictions.append(preds[batch_length].cpu().numpy().item())

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

    return accuracy, avgloss,class_acc,all_class_prec,all_predictions
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

if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    path = "..\\eurosat_dataset\\EuroSAT_RGB"
    mapped_data,class_names = load_data(path)
    seed = 42
    gen = torch.Generator()
    gen.manual_seed(seed)
    train_set,val_set,test_set = torch.utils.data.random_split(mapped_data,[0.7,0.15,0.15],generator=gen)
    # train_set, temp_data = train_test_split(mapped_data, train_size=0.70, test_size=0.3, random_state=seed)
    # val_set, test_set = train_test_split(temp_data, test_size=0.5, random_state=seed)
    # Testing
    with open('best_model.txt', 'rb') as filename:
        if(torch.cuda.is_available()):
            best_trf_model = pickle.load(filename)
        else:
            best_trf_model = CPU_Unpickler(filename).load()
    # get the best model used based on the transform IDX
    print("--------------Loading Model and Testing Phase--------------------")
    print(f"Best Model Transform IDX: {best_trf_model[0][2]}")
    print(f"Best Model Acc: {best_trf_model[0][1]}")

    test_dataset = CustomDataset(test_set, transform=transform_list[best_trf_model[0][2]][1])

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

    # load the best model chosen for testing
    checkpoint = torch.load(f'model\\best_model_transform{best_trf_model[0][2]+1}.pth',map_location=device)
    model = models.efficientnet_b0()
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 10)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    accuracy, _, test_class_acc, test_class_prec_acc,predictions = evaluate(model=model, dataloader=test_dataloader, criterion=None,
                                                                device=device)
    converted_preds = [class_names[pred_idx] for pred_idx in predictions]
    save_data(test_set,converted_preds)
    # idx_to_class = {predictions[i]: class_names[i] for i in range(len(predictions))}
    # converted_list = [idx_to_class[pred] for pred in predictions]
    # print(converted_list)
    print(f"Test Set Acc: {np.round(accuracy.item(), 2)}%")
    test_table = PrettyTable()
    test_table.field_names = ["Class", "Test Set Class Accuracy", "Test Set Class Precision"]
    for class_idx in range(len(test_class_acc)):
        test_table.add_row([f"{class_idx}", np.round(test_class_acc[class_idx]), np.round(test_class_prec_acc[class_idx])])
        # print(f"Test set Class {class_idx} acc: {np.round(test_class_acc[class_idx], 2)}%")
        # print(f"Test set Class {class_idx} precision: {np.round(test_class_prec_acc[class_idx], 2)}%")
    print(test_table)
    print(f'Test set Classes Average acc: {np.sum(test_class_acc) / len(test_class_acc)}')
    print(f'Test set Classes Average precision: {np.sum(test_class_prec_acc) / len(test_class_prec_acc)}')
    with open("test_results.txt", 'w') as file:
        file.write(f"Best Model Acc: {best_trf_model[0][1]}\n")
        file.write(f"Best Model Transform: {best_trf_model[0][2]+1}\n")
        file.write(f'\n\nTest Accuracy (per class)\n')
        for class_idx in range(len(test_class_acc)):
            file.write(f"Class {class_idx}: {test_class_acc[class_idx]}\n")
        file.write(f'\n\nTest AP Score (per class)\n')
        for class_idx in range(len(test_class_prec_acc)):
            file.write(f"Class {class_idx}: {test_class_prec_acc[class_idx]}\n")
        file.write(f'\n\nTest Mean AP Score Over All Classes: {np.sum(test_class_acc) / len(test_class_acc)}')
        file.write(f'\nTest Mean Accuracy Over All Classes: {np.sum(test_class_prec_acc) / len(test_class_prec_acc)}')
    print("Test results saved!")
    file.close()