
# coding: utf-8

# # VISUAL TRANSFORMER MODEL FOR COVID/NON-COVID CLASSIFICATION

__author__ = 'Ajay Arunachalam'
__date__ = '8.5.2021'

from __future__ import print_function
import glob
from itertools import chain
import os
import random
import zipfile
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from linformer import Linformer
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm.notebook import tqdm
import math
from vit_pytorch.efficient import ViT
import cv2
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix
import requests


print(f'Current Working Directory: {os.getcwd()}')


# ### Training settings
batch_size = 64
epochs = 20
lr = 3e-5
gamma = 0.7
seed = 142
IMG_SIZE = 224

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(seed)

device = 'cpu'

os.makedirs('lung_ct_scan_data', exist_ok=True)

train_dir = 'lung_ct_scan_data/train'
test_dir = 'lung_ct_scan_data/test'

train_list = glob.glob(os.path.join(train_dir,'*.png')) + glob.glob(os.path.join(train_dir,'*.jpg'))
test_list = glob.glob(os.path.join(test_dir, '*.png')) + glob.glob(os.path.join(test_dir, '*.jpg'))

print(f"Train Data: {len(train_list)}")
print(f"Test Data: {len(test_list)}")

labels = [path.split('\\')[-1].split(' (')[0] for path in train_list]

print(labels)

random_idx = np.random.randint(1, len(train_list), size=9)
fig, axes = plt.subplots(3, 3, figsize=(16, 12))

for idx, ax in enumerate(axes.ravel()):
    img = Image.open(train_list[idx])
    ax.set_title(labels[idx])
    ax.imshow(img)

train_list, valid_list = train_test_split(train_list, 
                                          test_size=0.2,
                                          stratify=labels,
                                          random_state=seed)
print(f"Train Data: {len(train_list)}")
print(f"Validation Data: {len(valid_list)}")
print(f"Test Data: {len(test_list)}")


# ### Image Augmentation

#Tune the transforms
ORIG_RES = False

if ORIG_RES:
    resize = 512
else:
    resize = 224

train_transforms = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        #transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)

val_transforms = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        #transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)


test_transforms = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        #transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)

# Fine tuning augmentation(comment the above code if using this one)
train_transforms = transforms.Compose(
     [transforms.RandomHorizontalFlip(),
     transforms.RandomRotation(10, resample=Image.BILINEAR),
    transforms.RandomAffine(8, translate=(.15,.15)),
    transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

val_transforms = transforms.Compose(
     [transforms.RandomHorizontalFlip(),
     transforms.RandomRotation(10, resample=Image.BILINEAR),
     transforms.RandomAffine(8, translate=(.15,.15)),
     transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

test_transforms = transforms.Compose(
     [transforms.RandomHorizontalFlip(),
     transforms.RandomRotation(10, resample=Image.BILINEAR),
     transforms.RandomAffine(8, translate=(.15,.15)),
     transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

class CovidNonCovidCTDataset(Dataset):
    def __init__(self, file_list, transform=None, orig_res=False):
        if orig_res:
            self.IMG_SIZE = 512
        else:
            self.IMG_SIZE = 224
        
        self.file_list = file_list
        self.transform = transform
        self.orig_res = orig_res
 
    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path).convert('RGB')
        img_transformed = self.transform(img)

        label = img_path.split("\\")[-1].split(" (")[0]
        label = 1 if label == "covid" else 0

        return img_transformed, label

train_data = CovidNonCovidCTDataset(train_list, transform=train_transforms, orig_res = ORIG_RES)
valid_data = CovidNonCovidCTDataset(valid_list, transform=test_transforms, orig_res = ORIG_RES)
test_data = CovidNonCovidCTDataset(test_list, transform=test_transforms, orig_res = ORIG_RES)

train_loader = DataLoader(dataset = train_data, batch_size=batch_size, shuffle=True )
valid_loader = DataLoader(dataset = valid_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset = test_data, batch_size=batch_size, shuffle=True)

print(len(train_data), len(train_loader))

print(len(valid_data), len(valid_loader))

# Effecient Attention

# Linformer

efficient_transformer = Linformer(
    dim=128,#128
    seq_len=49+1,  # 7x7 patches + 1 cls-token
    depth=12,
    heads=8,
    k=64
)

# Visual Transformer

model = ViT(
    dim=128,
    image_size=224,
    patch_size=32,
    num_classes=2,
    transformer=efficient_transformer,
   channels=3,
).to(device)


# loss function
criterion = nn.CrossEntropyLoss()
# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)
# scheduler
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

for epoch in range(epochs):
    epoch_loss = 0
    epoch_accuracy = 0
    for data, label in tqdm(train_loader):
        data = data.to(device)
        label = label.to(device)

        output = model(data)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = (output.argmax(dim=1) == label).float().mean()
        epoch_accuracy += acc / len(train_loader)
        epoch_loss += loss / len(train_loader)

        with torch.no_grad():
            epoch_val_accuracy = 0
            epoch_val_loss = 0
        for data, label in valid_loader:
            
            data = data.to(device)
            label = label.to(device)

            val_output = model(data)
            val_loss = criterion(val_output, label)

            acc = (val_output.argmax(dim=1) == label).float().mean()
            epoch_val_accuracy += acc / len(valid_loader)
            epoch_val_loss += val_loss / len(valid_loader)

    print(
        f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
    )

# Save model
PATH = ".\ViTModel.pt" # Use your own path
torch.save(model.state_dict(), PATH)

# performance on test data

def test(model, test_loader, criterion):
    '''
    Model testing 
    
    Args:
        model: model used during training and validation
        test_loader: data loader object containing testing data
        criterion: loss function used
    
    Returns:
        test_loss: calculated loss during testing
        accuracy: calculated accuracy during testing
        y_proba: predicted class probabilities
        y_truth: ground truth of testing data
    '''
    
    y_proba = []
    y_truth = []
    test_loss = 0
    total = 0
    correct = 0
    for data in tqdm(test_loader):
        X, y = data[0].to(device), data[1].to(device)
        output = model(X)
        test_loss += criterion(output, y.long()).item()
        for index, i in enumerate(output):
            y_proba.append(i[1])
            y_truth.append(y[index])
            if torch.argmax(i) == y[index]:
                correct+=1
            total+=1
                
    accuracy = correct/total
    
    y_proba_out = np.array([float(y_proba[i]) for i in range(len(y_proba))])
    y_truth_out = np.array([float(y_truth[i]) for i in range(len(y_truth))])
    
    return test_loss, accuracy, y_proba_out, y_truth_out


loss, acc, y_proba, y_truth = test(model, test_loader, criterion = nn.CrossEntropyLoss())


acc

pd.value_counts(y_truth)

### LOAD SAVED MODEL
# =============================================================================
# model = ViT()
# model.load_state_dict(torch.load(PATH))
# model.eval()            
# =============================================================================

def display_FP_FN(model, test_loader, criterion, display = 'fp'):
    '''
    Displaying false positive or false negative images.
    
    Args:
        model: model used during training, testing and validation
        test_loader: data loader object for testing set
        criterion: loss function used
        display: either 'fp' for displaying false positives or 'fn' for false negatives
    
    Returns: void
    '''
    
    fp = []
    fn = []
    for data in tqdm(test_loader):
        X, y = data[0].to(device), data[1].to(device)
        output = model(X)
        for index, i in enumerate(output):
            if torch.argmax(i.to(device)) == torch.Tensor([1]) and y[index].to(device) == torch.Tensor([0]):
                fp.append(X[index])
            elif torch.argmax(i.to(device)) == torch.Tensor([0]) and y[index].to(device) == torch.Tensor([1]):
                fn.append(X[index])
    
    fig = plt.figure()
    
    if display == 'fp':
        n_img = len(fp)
        cols = int(math.sqrt(n_img))
        for idx, img in enumerate(fp):
            a = fig.add_subplot(cols, np.ceil(n_img/float(cols)), idx + 1)
            plt.imshow(img.view(224, 224, 3).cpu())
            plt.axis('off')
    
            
    elif display == 'fn':
        n_img = len(fn)
        cols = int(math.sqrt(n_img))
        for idx, img in enumerate(fn):
            a = fig.add_subplot(cols, np.ceil(n_img/float(cols)), idx + 1)
            plt.imshow(img.view(224, 224, 3).cpu())
            plt.axis('off')

display_FP_FN(model, test_loader, criterion = nn.CrossEntropyLoss(), display = 'fn')

def get_confusion_matrix(y_truth, y_proba, labels):# labels
    '''
    Displays confusion matrix given output and ground truth data.
    
    Args:
        y_truth: ground truth for testing data output
        y_proba: class probabilties predicted from model
        labels: a list of labels for each cell of confusion matrix
    
    Returns:
        cm: returns a numpy array representing the confusion matrix
        
    '''
    
    y_in = np.array([round(i) for i in y_proba])
    cm = confusion_matrix(y_truth, y_in, labels)#, labels
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title('COVID Confusion Matrix')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    return cm

get_confusion_matrix(y_truth, y_proba, labels = [0, 1])#, labels = [0, 1]

def plot_ROCAUC_curve(y_truth, y_proba, fig_size):
    '''
    Plots the Receiver Operating Characteristic Curve (ROC) and displays Area Under the Curve (AUC) score.
    
    Args:
        y_truth: ground truth for testing data output
        y_proba: class probabilties predicted from model
        fig_size: size of the output pyplot figure
    
    Returns: void
    '''
    
    fpr, tpr, threshold = roc_curve(y_truth, y_proba)
    auc_score = roc_auc_score(y_truth, y_proba)
    txt_box = "AUC Score: " + str(round(auc_score, 4))
    plt.figure(figsize=fig_size)
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1],'--')
    plt.annotate(txt_box, xy=(0.65, 0.05), xycoords='axes fraction')
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")

plot_ROCAUC_curve(y_truth, y_proba, (8, 8))




