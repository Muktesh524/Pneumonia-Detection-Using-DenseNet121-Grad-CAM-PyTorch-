import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.cuda.amp import GradScaler, autocast
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
from pathlib import Path
import os
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class GradCAMPlusPlus:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []
        self.register_hooks()
    
    def register_hooks(self):
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        def forward_hook(module, input, output):
            self.activations = output
        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_backward_hook(backward_hook))
    
    def generate(self, input_tensor):
        self.model.eval()
        # Ensure input_tensor requires gradient
        if not input_tensor.requires_grad:
            input_tensor = input_tensor.clone().detach().requires_grad_(True).to(device)
        
        # Forward pass
        output = self.model(input_tensor)  # Shape: [1, 1] for binary classification
        score = output[0, 0]  # Use the single logit value
        self.model.zero_grad()
        
        # Backward pass
        score.backward(retain_graph=True)
        
        # Compute CAM
        gradients = self.gradients.data.cpu().numpy()  # Shape: [1, channels, H, W]
        activations = self.activations.data.cpu().numpy()  # Shape: [1, channels, H, W]
        
        # Compute weights as mean across spatial dimensions
        weights = np.mean(gradients[0], axis=(1, 2))  # Shape: [channels]
        
        # Initialize CAM
        cam = np.zeros(activations.shape[2:], dtype=np.float32)  # Shape: [H, W]
        
        # Weighted sum across channels
        for i in range(len(weights)):
            cam += weights[i] * activations[0, i, :, :]
        
        # Apply ReLU
        cam = np.maximum(cam, 0)
        
        # Resize to input image size
        cam = cv2.resize(cam, (224, 224))
        
        # Normalize
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)
        
        return cam

def setup_data_loaders(data_dir):
    print("Setting up data loaders...")
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val_test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    
    full_path = Path(data_dir)
    print(f"Looking for dataset at: {full_path}")
    
    if not full_path.exists():
        raise FileNotFoundError(f"Dataset not found at {full_path}. Please ensure 'train', 'val', and 'test' folders are inside '{data_dir}'.")
    
    full_train_dataset = datasets.ImageFolder(f"{full_path}/train", data_transforms['train'])
    val_dataset = datasets.ImageFolder(f"{full_path}/val", data_transforms['val_test'])
    test_dataset = datasets.ImageFolder(f"{full_path}/test", data_transforms['val_test'])
    
    if len(val_dataset) < 100:
        print("Val set too small; splitting from train for 70/15/15...")
        total_size = len(full_train_dataset)
        train_size = int(0.7 * total_size)
        val_size = int(0.15 * total_size)
        test_size = total_size - train_size - val_size
        train_dataset, _, _ = random_split(full_train_dataset, [train_size, val_size, test_size])
        val_dataset.dataset = datasets.ImageFolder(f"{full_path}/val", data_transforms['val_test'])
    else:
        train_dataset = full_train_dataset
    
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True),
        'val': DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True),
        'test': DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)
    }
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val', 'test']}
    class_names = full_train_dataset.classes
    print(f"Dataset sizes: {dataset_sizes}")
    print(f"Classes: {class_names}")
    return dataloaders, dataset_sizes, class_names

def setup_model():
    print("Setting up DenseNet121...")
    model = models.densenet121(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, 1)  # Binary classification
    model = model.to(device)
    return model

def train_model(model, dataloaders, dataset_sizes, epochs=10, lr=0.0001):
    print(f"Training for {epochs} epochs...")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    scaler = GradScaler()
    best_acc = 0.0
    
    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            optimizer.zero_grad()
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / dataset_sizes['train']
        model.eval()
        correct = 0
        with torch.no_grad():
            for inputs, labels in dataloaders['val']:
                inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
                with autocast():
                    outputs = model(inputs)
                    preds = (torch.sigmoid(outputs) > 0.5).float()
                correct += (preds == labels).sum().item()
        acc = correct / dataset_sizes['val']
        print(f'Loss: {epoch_loss:.4f} Acc: {acc:.4f}')
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'outputs/models/best_densenet121.pth')
        scheduler.step()
    return model

def fine_tune_model(model, dataloaders, dataset_sizes, epochs=10, lr=0.00001):
    print(f"Fine-tuning for {epochs} epochs...")
    for param in model.parameters():
        param.requires_grad = True
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    scaler = GradScaler()
    best_acc = 0.0
    patience = 0
    
    for epoch in range(epochs):
        print(f'Fine-tune Epoch {epoch+1}/{epochs}')
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            optimizer.zero_grad()
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / dataset_sizes['train']
        model.eval()
        correct = 0
        with torch.no_grad():
            for inputs, labels in dataloaders['val']:
                inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
                with autocast():
                    outputs = model(inputs)
                    preds = (torch.sigmoid(outputs) > 0.5).float()
                correct += (preds == labels).sum().item()
        acc = correct / dataset_sizes['val']
        print(f'Loss: {epoch_loss:.4f} Acc: {acc:.4f}')
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'outputs/models/finetuned_densenet121.pth')
            patience = 0
        else:
            patience += 1
        if patience >= 5:
            print("Early stopping triggered")
            break
        scheduler.step()
    return model

def evaluate_model(model, dataloader, dataset_size, class_names):
    print("Evaluating model...")
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            with autocast():
                outputs = model(inputs)
                preds = (torch.sigmoid(outputs) > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.savefig('outputs/logs/confusion_matrix.png')
    plt.close()
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}, cm

def visualize_gradcam(model, dataloader, class_names):
    print("Generating Grad-CAM++ heatmaps...")
    target_layer = model.features[-1]
    gradcam = GradCAMPlusPlus(model, target_layer)
    model.eval()
    images_shown = 0
    for inputs, labels in dataloader:
        for i in range(inputs.size(0)):
            if images_shown >= 5:
                break
            # Create input tensor with gradient tracking
            input_tensor = inputs[i:i+1].clone().detach().requires_grad_(True).to(device)
            label = labels[i].item()
            true_class = class_names[label]
            with autocast():
                output = model(input_tensor)
                pred_idx = 1 if torch.sigmoid(output) > 0.5 else 0
            pred_class = class_names[pred_idx]
            cam = gradcam.generate(input_tensor)
            img = input_tensor[0].cpu().detach().permute(1, 2, 0).numpy()
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = np.clip(img, 0, 1)
            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
            superimposed = heatmap * 0.4 + img * 0.6
            plt.figure(figsize=(15, 5))
            plt.subplot(131), plt.imshow(img), plt.title(f'True: {true_class}'), plt.axis('off')
            plt.subplot(132), plt.imshow(cam, cmap='jet'), plt.title('Grad-CAM++'), plt.axis('off')
            plt.subplot(133), plt.imshow(superimposed), plt.title(f'Pred: {pred_class}'), plt.axis('off')
            plt.savefig(f'outputs/heatmaps/sample_{images_shown+1}.png')
            plt.close()
            images_shown += 1
        if images_shown >= 5:
            break

def main():
    print("Pneumonia Detection Project - Team 4")
    data_dir = "data/DATASET"
    os.makedirs("outputs/models", exist_ok=True)
    os.makedirs("outputs/logs", exist_ok=True)
    os.makedirs("outputs/heatmaps", exist_ok=True)
    
    if not torch.cuda.is_available():
        print("Warning: CUDA is not available. Training will use CPU. Please verify GPU setup.")
    dataloaders, dataset_sizes, class_names = setup_data_loaders(data_dir)
    model = setup_model()
    model = train_model(model, dataloaders, dataset_sizes, epochs=10)
    model = fine_tune_model(model, dataloaders, dataset_sizes, epochs=10)
    metrics, cm = evaluate_model(model, dataloaders['test'], dataset_sizes['test'], class_names)
    visualize_gradcam(model, dataloaders['test'], class_names)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'outputs/logs/report_{timestamp}.txt', 'w') as f:
        f.write(f"Pneumonia Detection Report\nDate: {datetime.now()}\nMetrics: {metrics}\n")
    print("Project completed! Check outputs/ for results.")

if __name__ == "__main__":
    main()