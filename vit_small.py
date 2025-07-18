import torch
from torchvision import datasets, transforms
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import copy
from tqdm import tqdm
import os
import logging
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from collections import Counter
from multiclass_classifiers import MODEL_TRAINING

# logging setup
os.makedirs('/home/swarnendu-sengupta/Work_NO_SHARE/SARIFA_multi_class/logs', exist_ok=True)
log_file = '/home/swarnendu-sengupta/Work_NO_SHARE/SARIFA_multi_class/logs/vitBest_training.log'
logging.basicConfig(
        filename=log_file,
        filemode='w',
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO
    )
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

# Enhanced transforms with stronger augmentation
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
    transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))], p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.1))
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Split dataset into train, validation, and test sets
dataset_folder = '/home/swarnendu-sengupta/Work_NO_SHARE/SARIFA_multi_class/NCT-CRC-HE-100K-NONORM'
dataset = datasets.ImageFolder(dataset_folder)
targets = dataset.targets

logging.info(f"Dataset size: {len(dataset)}")
logging.info(f"Classes: {dataset.classes}")
# Print total number of parameters
logging.info(dataset.class_to_idx)  # Shows the mapping from class name to label
logging.info(dataset.classes)       # List of class names in order of their label


indices = np.arange(len(targets))
train_indices, val_test_indices = train_test_split(indices, test_size=0.4, random_state=42, stratify=targets)
temp_targets = np.array(targets)[val_test_indices]
val_indices, test_indices = train_test_split(val_test_indices, test_size=0.5, random_state=42, stratify=temp_targets)

train_dataset = datasets.ImageFolder(dataset_folder, transform = train_transform)
test_dataset = datasets.ImageFolder(dataset_folder, transform = val_test_transform)
val_dataset = datasets.ImageFolder(dataset_folder, transform = val_test_transform)

train_dataset = Subset(train_dataset, train_indices)
val_dataset = Subset(val_dataset, val_indices)
test_dataset = Subset(test_dataset, test_indices)

print(np.bincount(np.array(targets)[train_indices]))
print(np.bincount(np.array(targets)[val_indices]))
print(np.bincount(np.array(targets)[test_indices]))

train_labels = np.array(targets)[train_indices]

model_save_path = '/home/swarnendu-sengupta/Work_NO_SHARE/SARIFA_multi_class/models/'
os.makedirs(model_save_path, exist_ok =True)

MODEL_TRAINING(train_dataset = train_dataset, val_dataset = val_dataset, test_dataset = test_dataset,
            class_names = dataset.classes, train_class_labels = train_labels, log_file = log_file, save_path = model_save_path,
                  model_type='vit_small', learning_rate=1e-3, num_epochs=500, batch_size=256)
