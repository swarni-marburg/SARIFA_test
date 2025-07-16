import copy
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import warnings
import random
from collections import Counter
import timm
import json
from datetime import datetime
warnings.filterwarnings('ignore')

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def calculate_topk_accuracy(outputs, targets, k=3):
    """Calculate top-k accuracy"""
    with torch.no_grad():
        batch_size = targets.size(0)
        _, pred = outputs.topk(k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        
        topk_accuracies = []
        for i in range(1, k+1):
            correct_k = correct[:i].contiguous().view(-1).float().sum(0, keepdim=True)
            topk_accuracies.append(correct_k.mul_(100.0 / batch_size).item())
        
        return topk_accuracies

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class VisionTransformerL(nn.Module):
    """Enhanced Vision Transformer with better architecture for medical images"""
    
    def __init__(self, num_classes=9, model_name='vitl', pretrained=True):
        super().__init__()
        
        # Use timm's vision transformer
        self.backbone = models.vit_l_16(weights='ViT_L_16_Weights.IMAGENET1K_SWAG_LINEAR_V1')

        # Remove the default classification head
        self.backbone.heads = nn.Identity()

        # Freeze all parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze only the last transformer encoder block
        # For timm ViT, the transformer blocks are usually in model.blocks
        for param in self.backbone.encoder.layers[-1].parameters():
            param.requires_grad = True
        
        # Get feature dimension
        self.feature_dim = self.backbone.encoder.ln.normalized_shape[0]
        
        # Enhanced classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
        
        # Initialize classifier weights
        self._init_weights()
        
    def _init_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


class EnhancedVisionTransformer(nn.Module):
    """Enhanced Vision Transformer with better architecture for medical images"""
    
    def __init__(self, num_classes=6, model_name='vit_base_patch16_224', pretrained=True):
        super().__init__()
        
        # Use timm's vision transformer
        self.backbone = timm.create_model(model_name, pretrained=pretrained)

        # Remove the default classification head
        self.backbone.head = nn.Identity()

        # Freeze all parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze only the last transformer encoder block
        # For timm ViT, the transformer blocks are usually in model.blocks
        for param in self.backbone.blocks[-1].parameters():
            param.requires_grad = True
        
        # Get feature dimension
        self.feature_dim = self.backbone.num_features
        
        # Enhanced classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
        
        # Initialize classifier weights
        self._init_weights()
        
    def _init_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

class ViTBest(nn.Module):
    def __init__(self, num_classes=9, model_name='vit_best', pretrained=True):
        super().__init__()
        
        # Use timm's EfficientNet
        self.backbone = timm.create_model('caformer_b36.sail_in22k_ft_in1k', pretrained=True)

        # Get feature dimension
        self.feature_dim = self.backbone.head.fc.fc1.in_features

        # Remove the default classification head
        self.backbone.head.fc = nn.Identity()


        # Freeze all parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
               
        # Enhanced classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
        
        # Initialize classifier weights
        self._init_weights()
        
    def _init_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

class ViTConvformer(nn.Module):
    def __init__(self, num_classes=9, model_name='vit_convformer', pretrained=True):
        super().__init__()
        
        # Use timm's EfficientNet
        self.backbone = timm.create_model('convformer_b36.sail_in22k_ft_in1k', pretrained=True)

        # Get feature dimension
        self.feature_dim = self.backbone.head.fc.fc1.in_features

        # Remove the default classification head
        self.backbone.head.fc = nn.Identity()


        # Freeze all parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
               
        # Enhanced classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
        
        # Initialize classifier weights
        self._init_weights()
        
    def _init_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


class EarlyStopping:
    def __init__(self, path, model_type, patience=7, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.best_model_wts = None
        self.save_path = os.path.join(path, model_type+'_best_model.pth')

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), self.save_path)
            logging.info(f"Validation loss improved to {val_loss:.4f}, saving model weights to {self.save_path}")
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def MODEL_TRAINING(train_dataset, val_dataset, test_dataset, class_names, train_class_labels, log_file, save_path,
                  model_type='vit', learning_rate=1e-4, num_epochs=50, batch_size=16):

    # List of class names in the order you want to use for reporting
    classes = ['BACK','DEB','LYM','MUC','MUS','NORM','STR','ADI','TUM']

    """Enhanced training with class imbalance handling and top-k accuracy"""
    logging.basicConfig(
        filename=log_file,
        filemode='a',
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO
        )


    logging.info(f"Starting enhanced model training with {model_type}...")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=32, pin_memory=True, drop_last = True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    
    # Initialize model
    if model_type == 'vitb':
        model = EnhancedVisionTransformer(num_classes=len(class_names))
    elif model_type == 'efficientnet':
        model = EfficientNetClassifier(num_classes=len(class_names))
    elif model_type == 'vitl':
        model = VisionTransformerL(num_classes = len(class_names))
    elif model_type == 'vit_best':
        model = ViTBest(num_classes = len(class_names))
    elif model_type == 'vit_convformer':
        model = ViTConvformer(num_classes = len(class_names))
    else:
        raise ValueError("model_type must be 'vit' or 'efficientnet'")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    torch.cuda.empty_cache()
    
    # Calculate class weights for focal loss
    train_class_counts = Counter(train_class_labels)
    class_counts = list(train_class_counts.values())
    class_weights = compute_class_weight('balanced', classes=np.arange(len(class_counts)), 
                                       y=train_class_labels)
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    # Use Focal Loss instead of CrossEntropy
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    
    # Enhanced optimizer with different learning rates
    optimizer = optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': learning_rate * 0.1},  # Lower LR for backbone
        {'params': model.classifier.parameters(), 'lr': learning_rate}       # Higher LR for classifier
    ], weight_decay=0.01)
    
    # Enhanced scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    # Training metrics
    train_losses = []
    train_accuracies = []
    val_accuracies = []
    val_f1_scores = []
    val_top1_accuracies = []
    val_top2_accuracies = []
    val_top3_accuracies = []
    best_val_f1 = 0.0
    
    logging.info(f"Trainable Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    early_stopping = EarlyStopping(model_type = model_type, path = save_path, patience=10, min_delta=0.0001)
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        val_loss = 0.0
        train_correct = 0
        train_total = 0
        train_preds = []
        train_labels = []
        val_predictions = []
        val_targets = []
        for data, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item() * data.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            train_preds.extend(predicted.cpu().numpy())
            train_labels.extend(targets.cpu().numpy())
        
        # Compute training metrics
        train_cm = confusion_matrix(train_labels, train_preds, labels=list(range(len(classes))))
        train_recall = recall_score(train_labels, train_preds, average=None, zero_division=0)
        train_precision = precision_score(train_labels, train_preds, average=None, zero_division=0)
        train_f1 = f1_score(train_labels, train_preds, average=None, zero_division=0)
        train_per_class_acc = train_cm.diagonal() / train_cm.sum(axis=1)
        train_acc = sum(train_cm.diagonal())/sum(sum(train_cm))
        train_topk_accs = calculate_topk_accuracy(train_preds, train_labels, k=3)
        print(f'Top k acc : {train_topk_accs}')
        train_specificity = []
        for i in range(len(classes)):
            TP = train_cm[i, i]
            FP = train_cm[:, i].sum() - TP
            FN = train_cm[i, :].sum() - TP
            TN = train_cm.sum() - (TP + FP + FN)
            train_specificity.append(TN / (TN + FP) if (TN + FP) > 0 else 0)

        # Validation phase
        model.eval()
        val_predictions = []
        val_targets = []
        val_top1_acc = 0
        val_top2_acc = 0
        val_top3_acc = 0
        val_batches = 0
        
        with torch.no_grad():
            for data, targets in tqdm(val_loader, desc ='Validation'):
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs, 1)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * data.size(0)
                val_predictions.extend(predicted.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())
                
                # Calculate top-k accuracies
                topk_accs = calculate_topk_accuracy(outputs, targets, k=3)
                val_top1_acc += topk_accs[0]
                val_top2_acc += topk_accs[1]
                val_top3_acc += topk_accs[2]
                val_batches += 1
        
        # Calculate metrics
        epoch_train_loss = train_loss / len(train_loader)
        epoch_train_acc = 100. * train_correct / train_total
        epoch_val_acc = 100. * accuracy_score(val_targets, val_predictions)
        epoch_val_f1 = f1_score(val_targets, val_predictions, average='weighted')
        epoch_val_top1 = val_top1_acc / val_batches
        epoch_val_top2 = val_top2_acc / val_batches
        epoch_val_top3 = val_top3_acc / val_batches
        
        # Compute validation metrics
        val_cm = confusion_matrix(val_targets, val_predictions, labels=list(range(len(classes))))
        val_recall = recall_score(val_targets, val_predictions, average=None, zero_division=0)
        val_precision = precision_score(val_targets, val_predictions, average=None, zero_division=0)
        val_f1 = f1_score(val_targets, val_predictions, average=None, zero_division=0)
        val_per_class_acc = val_cm.diagonal() / val_cm.sum(axis=1)  # Per-class accuracy[2][5][9]
        val_acc = sum(val_cm.diagonal())/sum(sum(val_cm))
        val_specificity = []
        for i in range(len(classes)):
            TP = val_cm[i, i]
            FP = val_cm[:, i].sum() - TP
            FN = val_cm[i, :].sum() - TP
            TN = val_cm.sum() - (TP + FP + FN)
            val_specificity.append(TN / (TN + FP) if (TN + FP) > 0 else 0)

        logging.info(f"Epoch [{epoch+1}/{num_epochs}]")
        logging.info(f"  Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}")
        for idx, cls in enumerate(classes):
            logging.info(f"    [Train] {cls:>4s} | Acc: {train_per_class_acc[idx]:.4f} | Sensitivity: {train_recall[idx]:.4f} | Specificity: {train_specificity[idx]:.4f} | F1: {train_f1[idx]:.4f}")
        logging.info(f"  Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}")
        for idx, cls in enumerate(classes):
            logging.info(f"    [Val]   {cls:>4s} | Acc: {val_per_class_acc[idx]:.4f} | Sensitivity: {val_recall[idx]:.4f} | Specificity: {val_specificity[idx]:.4f} | F1: {val_f1[idx]:.4f}")


        # Early stopping check
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            logging.info(f"Early stopping at epoch {epoch+1}")
            # Restore best model weights
            model.load_state_dict(early_stopping.best_model_wts)

            # Save model info
            model_info = {
                'epoch': epoch + 1,
                'model_type': model_type,
                'best_val_f1': best_val_f1,
                'val_accuracy': epoch_val_acc,
                'class_names': class_names,
                'num_classes': len(class_names),
                'timestamp': datetime.now().isoformat(),
                'train_loss': epoch_train_loss,
                'train_accuracy': epoch_train_acc,
                'val_loss': epoch_val_f1,
                'val_accuracy': epoch_val_acc,
                'val_top1_accuracy': epoch_val_top1,
                'val_top2_accuracy': epoch_val_top2,
                'val_top3_accuracy': epoch_val_top3,
                'train_class_counts': train_class_counts,
                'train_per_class_accuracy': train_per_class_acc.tolist(),
                'val_per_class_accuracy': val_per_class_acc.tolist(),
                'train_recall': train_recall.tolist(),
                'val_recall': val_recall.tolist(),
                'train_precision': train_precision.tolist(),
                'val_precision': val_precision.tolist(),
                'train_f1': train_f1.tolist(),
                'val_f1': val_f1.tolist(),
                'train_specificity': train_specificity,
                'val_specificity': val_specificity,
                'train_cm': train_cm.tolist(),
                'val_cm': val_cm.tolist(),
            }
            
            with open(os.path.join(save_path,f'best_{model_type}_model_info.json'), 'w') as f:
                json.dump(model_info, f, indent=2)

            break

        logging.info(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        scheduler.step()  # Step the learning rate scheduler after each epoch

        
        # Store metrics
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)
        val_accuracies.append(epoch_val_acc)
        val_f1_scores.append(epoch_val_f1)
        val_top1_accuracies.append(epoch_val_top1)
        val_top2_accuracies.append(epoch_val_top2)
        val_top3_accuracies.append(epoch_val_top3)
        
        logging.info(f'Epoch {epoch+1}/{num_epochs}:')
        logging.info(f'  Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%')
        logging.info(f'  Val Acc: {epoch_val_acc:.2f}%, Val F1: {epoch_val_f1:.4f} (Best F1: {best_val_f1:.4f})')
        logging.info(f'  Val Top-1: {epoch_val_top1:.2f}%, Top-2: {epoch_val_top2:.2f}%, Top-3: {epoch_val_top3:.2f}%')
        logging.info(f'  Learning Rate: {scheduler.get_last_lr()[0]:.2e}')
        logging.info('-' * 50)
        
    return model, train_losses, train_accuracies, val_accuracies, val_f1_scores, val_top1_accuracies, val_top2_accuracies, val_top3_accuracies


def get_tta_transforms(image_size=(224, 224), n_tta=10):
    """Generate TTA transforms"""
    tta_transforms = []
    
    for i in range(n_tta):
        tta_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(image_size, scale=(0.9, 1.0)),
            transforms.RandomRotation(degrees=10),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        tta_transforms.append(tta_transform)
    
    return tta_transforms

def evaluate_model_with_tta(model, test_dataset, class_names, n_tta=10):
    """Evaluate model with Test Time Augmentation and top-k accuracy"""
    model.eval()
    
    tta_transforms = get_tta_transforms(n_tta=n_tta)
    
    all_predictions = []
    all_targets = []
    all_outputs = []
    
    logging.info(f"Evaluating with TTA (n_tta={n_tta})...")
    
    with torch.no_grad():
        for idx in tqdm(range(len(test_dataset)), desc='TTA Evaluation'):
            image_path = test_dataset.image_paths[idx]
            label = test_dataset.labels[idx]
            
            # Load original image
            image = Image.open(image_path).convert('RGB')
            
            # Apply TTA transforms and get predictions
            tta_outputs = []
            for tta_transform in tta_transforms:
                augmented_image = tta_transform(image).unsqueeze(0).to(device)
                output = model(augmented_image)
                tta_outputs.append(F.softmax(output, dim=1))
            
            # Average predictions across all TTA
            avg_output = torch.mean(torch.stack(tta_outputs), dim=0)
            _, predicted = avg_output.max(1)
            
            all_predictions.append(predicted.cpu().numpy()[0])
            all_targets.append(label)
            all_outputs.append(avg_output.cpu().numpy()[0])
    
    return all_predictions, all_targets, all_outputs

def evaluate_model_enhanced(model, test_loader, class_names):
    """Enhanced model evaluation with detailed metrics including top-k accuracy"""
    model.eval()
    all_predictions = []
    all_targets = []
    all_outputs = []
    
    with torch.no_grad():
        for data, targets in tqdm(test_loader, desc='Evaluating'):
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_outputs.extend(outputs.cpu().numpy())
    
    return all_predictions, all_targets, all_outputs

def calculate_topk_accuracy_from_outputs(outputs, targets, k=3):
    """Calculate top-k accuracy from outputs and targets"""
    outputs = torch.tensor(outputs)
    targets = torch.tensor(targets)
    
    # Get top-k predictions
    _, pred = outputs.topk(k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    
    topk_accuracies = []
    for i in range(1, k+1):
        correct_k = correct[:i].contiguous().view(-1).float().sum(0)
        accuracy = correct_k.mul_(100.0 / targets.size(0)).item()
        topk_accuracies.append(accuracy)
    
    return topk_accuracies

def print_evaluation_results(predictions, targets, class_names, method_name, outputs=None):
    """Print detailed evaluation results including top-k accuracy"""
    accuracy = accuracy_score(targets, predictions)
    f1_weighted = f1_score(targets, predictions, average='weighted')
    f1_macro = f1_score(targets, predictions, average='macro')
    
    logging.info(f"\n{method_name} Results:")
    logging.info(f"Top-1 Accuracy: {accuracy:.4f}")
    logging.info(f"Weighted F1-Score: {f1_weighted:.4f}")
    logging.info(f"Macro F1-Score: {f1_macro:.4f}")
    
    # Calculate top-k accuracies if outputs are provided
    if outputs is not None:
        topk_accs = calculate_topk_accuracy_from_outputs(outputs, targets, k=3)
        logging.info(f"Top-1 Accuracy: {topk_accs[0]:.2f}%")
        logging.info(f"Top-2 Accuracy: {topk_accs[1]:.2f}%")
        logging.info(f"Top-3 Accuracy: {topk_accs[2]:.2f}%")
    
    logging.info(f"\nDetailed Classification Report ({method_name}):")
    logging.info(classification_report(targets, predictions, target_names=class_names, digits=4))
    
    return accuracy, f1_weighted, f1_macro
'''
def plot_confusion_matrices(predictions_normal, predictions_tta, targets, class_names):
    """Plot confusion matrices for both methods"""
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # Normal confusion matrix (raw)
    cm_normal = confusion_matrix(targets, predictions_normal)
    sns.heatmap(cm_normal, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=axes[0,0])
    axes[0,0].set_title('Confusion Matrix - Normal (Raw Counts)')
    axes[0,0].set_ylabel('True Label')
    axes[0,0].set_xlabel('Predicted Label')
    
    # Normal confusion matrix (normalized)
    cm_normal_norm = cm_normal.astype('float') / cm_normal.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normal_norm, annot=True, fmt='.3f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=axes[0,1])
    axes[0,1].set_title('Confusion Matrix - Normal (Normalized)')
    axes[0,1].set_ylabel('True Label')
    axes[0,1].set_xlabel('Predicted Label')
    
    # TTA confusion matrix (raw)
    cm_tta = confusion_matrix(targets, predictions_tta)
    sns.heatmap(cm_tta, annot=True, fmt='d', cmap='Greens', 
                xticklabels=class_names, yticklabels=class_names, ax=axes[1,0])
    axes[1,0].set_title('Confusion Matrix - TTA (Raw Counts)')
    axes[1,0].set_ylabel('True Label')
    axes[1,0].set_xlabel('Predicted Label')
    
    # TTA confusion matrix (normalized)
    cm_tta_norm = cm_tta.astype('float') / cm_tta.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_tta_norm, annot=True, fmt='.3f', cmap='Greens', 
                xticklabels=class_names, yticklabels=class_names, ax=axes[1,1])
    axes[1,1].set_title('Confusion Matrix - TTA (Normalized)')
    axes[1,1].set_ylabel('True Label')
    axes[1,1].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.show()

def plot_enhanced_training_history(train_losses, train_accuracies, val_accuracies, val_f1_scores, val_top1_accuracies, val_top2_accuracies, val_top3_accuracies):
    """Enhanced training history visualization with top-k accuracy"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Training loss
    ax1.plot(train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.set_title('Training Loss', fontsize=14)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracies
    ax2.plot(train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Model Accuracy', fontsize=14)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # F1 Scores
    ax3.plot(val_f1_scores, 'g-', label='Validation F1 Score (Weighted)', linewidth=2)
    ax3.set_title('Validation F1 Score', fontsize=14)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('F1 Score')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Top-k Accuracies
    ax4.plot(val_top1_accuracies, 'b-', label='Top-1 Accuracy', linewidth=2)
    ax4.plot(val_top2_accuracies, 'g-', label='Top-2 Accuracy', linewidth=2)
    ax4.plot(val_top3_accuracies, 'r-', label='Top-3 Accuracy', linewidth=2)
    ax4.set_title('Validation Top-k Accuracy', fontsize=14)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy (%)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    '''
