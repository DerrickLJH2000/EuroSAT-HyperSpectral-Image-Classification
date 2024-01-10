import os
from PIL import Image
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder


# Function for Custom Dataset
class CustomDataset():
    def __init__(self, data_list, transform):
        self.data_list = data_list
        self.label_encoder = LabelEncoder()
        self.label_binarizer = LabelBinarizer()
        self.transform = transform

        labels = [label for _, label in self.data_list]

        # Updated: Binarize labels for multi-label classification
        self.labels = self.label_binarizer.fit_transform(labels)
        self.label_encoder.fit(labels)

    def __len__(self):
        return len(self.data_list)
        
    def __getitem__(self, idx):
        img_path, label = self.data_list[idx]
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)


        labels = self.labels[idx]
        if labels[0] == 1:      # if AnnualCrop == 1, PermanentCrop == 1
            labels[6] = 1
        elif labels[6] == 1:    # if PermanentCrop == 1, AnnualCrop == 1
            labels[0] = 1
        elif labels[1] == 1:    # if Forest == 1, HerbaceousVegetation == 1
            labels[2] = 1
        return img, labels


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

