#!/usr/bin/env python3
"""
PatchCore for Welding Anomaly Detection
Adapted from the original patchcore_example.py to work with extracted welding patches.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from torchvision import transforms
from torchvision.models import resnet50
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
from tqdm import tqdm
import warnings

# warnings.filterwarnings("ignore")

class WeldingDataset(Dataset):
    """Dataset for welding patch images"""
    
    def __init__(self, data_dir, transform=None, class_type='good'):
        """
        Initialize dataset
        
        Args:
            data_dir (str): Directory containing welding patches
            transform: Torchvision transforms
            class_type (str): 'good', 'bad', or 'all'
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.class_type = class_type
        
        # Get image paths
        class_dir = self.data_dir / class_type
        if not class_dir.exists():
            raise ValueError(f"Directory {class_dir} does not exist")
            
        self.image_paths = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.jpeg')) + list(class_dir.glob('*.png'))
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {class_dir}")
            
        print(f"Found {len(self.image_paths)} images in {class_dir}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Load image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL for transforms
        from PIL import Image
        image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
            
        return {
            'sample': image,
            'image_path': str(img_path),
            'label': 0 if self.class_type == 'good' else 1
        }

class Feature_extractor(nn.Module):
    """Feature extractor using ResNet50"""
    
    def __init__(self):
        super(Feature_extractor, self).__init__()
        self.model = resnet50(weights="DEFAULT")                         
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        def hook(model, input, output):
            self.features.append(output)
        
        self.model.layer2[-1].register_forward_hook(hook)
        self.model.layer3[-1].register_forward_hook(hook)

    def forward(self, x):
        self.features = []
        
        with torch.no_grad():
            _ = self.model(x)
        
        self.avg = nn.AvgPool2d(3, stride=1)
        self.shape = self.features[0].shape[-2]
        self.resize = nn.AdaptiveAvgPool2d(self.shape)

        resized_patches = [self.resize(self.avg(f)) for f in self.features]
        resized_patches = torch.cat(resized_patches, dim=1)
        patches = resized_patches.reshape(resized_patches.shape[1], -1).T

        return patches

class WeldingPatchCore:
    """PatchCore for welding anomaly detection"""
    
    def __init__(self, train_data_dir, test_data_dir=None, device=None):
        """
        Initialize PatchCore for welding
        
        Args:
            train_data_dir (str): Directory containing training welding patches
            test_data_dir (str): Directory containing test welding patches
            device: Torch device
        """
        self.train_data_dir = Path(train_data_dir)
        self.test_data_dir = Path(test_data_dir) if test_data_dir else None
        
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Transform for 224x224 input (ResNet50 compatible)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.feature_extractor = Feature_extractor().to(self.device)
        self.memory_bank = None
        self.threshold = None
        
    def build_memory_bank(self, subsample_ratio=0.1):
        """
        Build memory bank from good welding patches
        
        Args:
            subsample_ratio (float): Ratio of patches to keep in memory bank
        """
        print("Building memory bank from good welding patches...")
        
        # Load good welding patches
        good_dataset = WeldingDataset(
            self.train_data_dir, 
            transform=self.transform, 
            class_type='good'
        )
        
        dataloader = DataLoader(good_dataset, batch_size=32, shuffle=False)
        
        features_list = []
        
        for batch in tqdm(dataloader, desc="Extracting features"):
            images = batch['sample'].to(self.device)
            
            with torch.no_grad():
                features = self.feature_extractor(images)
                features_list.append(features.cpu())
        
        # Concatenate all features
        all_features = torch.cat(features_list, dim=0)
        
        # Subsample for memory bank
        n_samples = int(all_features.shape[0] * subsample_ratio)
        if n_samples < 100:  # Ensure minimum samples
            n_samples = min(100, all_features.shape[0])
            
        indices = np.random.choice(all_features.shape[0], size=n_samples, replace=False)
        self.memory_bank = all_features[indices].to(self.device)
        
        print(f"Memory bank built with {self.memory_bank.shape[0]} patches")
        
    def compute_threshold(self):
        """Compute anomaly threshold from training data"""
        print("Computing anomaly threshold...")
        
        good_dataset = WeldingDataset(
            self.train_data_dir,
            transform=self.transform,
            class_type='good'
        )
        
        dataloader = DataLoader(good_dataset, batch_size=32, shuffle=False)
        scores = []
        
        for batch in tqdm(dataloader, desc="Computing training scores"):
            images = batch['sample'].to(self.device)
            
            with torch.no_grad():
                features = self.feature_extractor(images)
                distances = torch.cdist(features, self.memory_bank)
                min_distances, _ = torch.min(distances, dim=1)
                max_score = min_distances.max().item()
                scores.append(max_score)
        
        # Set threshold as mean + 2*std
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        self.threshold = mean_score + 2 * std_score
        
        print(f"Computed threshold: {self.threshold:.4f}")
        print(f"Training score stats - Mean: {mean_score:.4f}, Std: {std_score:.4f}")
        
        return scores
        
    def predict(self, data_dir):
        """
        Predict anomalies in test data
        
        Args:
            data_dir (str): Directory containing test images
            
        Returns:
            tuple: (scores, predictions, true_labels, image_paths)
        """
        if self.memory_bank is None:
            raise ValueError("Memory bank not built. Call build_memory_bank() first.")
        if self.threshold is None:
            raise ValueError("Threshold not computed. Call compute_threshold() first.")
            
        print("Running inference on test data...")
        
        # Load both good and bad patches for testing
        all_scores = []
        all_predictions = []
        all_labels = []
        all_paths = []
        
        for class_type in ['good', 'bad']:
            class_dir = Path(data_dir) / class_type
            if not class_dir.exists():
                print(f"Warning: {class_dir} does not exist, skipping...")
                continue
                
            dataset = WeldingDataset(
                data_dir,
                transform=self.transform,
                class_type=class_type
            )
            
            dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
            
            for batch in tqdm(dataloader, desc=f"Predicting {class_type}"):
                images = batch['sample'].to(self.device)
                paths = batch['image_path']
                labels = batch['label'].numpy()
                
                with torch.no_grad():
                    features = self.feature_extractor(images)
                    distances = torch.cdist(features, self.memory_bank)
                    min_distances, _ = torch.min(distances, dim=1)
                    scores = min_distances.max(dim=0)[0].cpu().numpy()
                    
                    predictions = (scores > self.threshold).astype(int)
                    
                    all_scores.extend(scores)
                    all_predictions.extend(predictions)
                    all_labels.extend(labels)
                    all_paths.extend(paths)
        
        return all_scores, all_predictions, all_labels, all_paths
    
    def evaluate(self, test_data_dir, save_dir=None):
        """
        Evaluate model on test data
        
        Args:
            test_data_dir (str): Directory containing test images
            save_dir (str): Directory to save results
        """
        scores, predictions, labels, paths = self.predict(test_data_dir)
        
        # Calculate metrics
        auc_score = roc_auc_score(labels, scores)
        
        print(f"\n=== Evaluation Results ===")
        print(f"AUC-ROC Score: {auc_score:.4f}")
        print(f"Threshold used: {self.threshold:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(labels, predictions, target_names=['Good', 'Bad']))
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        print("\nConfusion Matrix:")
        print("Predicted:  Good  Bad")
        print(f"Good:       {cm[0,0]:4d}  {cm[0,1]:3d}")
        print(f"Bad:        {cm[1,0]:4d}  {cm[1,1]:3d}")
        
        # Save results
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Save metrics
            with open(save_path / 'metrics.txt', 'w') as f:
                f.write(f"AUC-ROC Score: {auc_score:.4f}\n")
                f.write(f"Threshold: {self.threshold:.4f}\n")
                f.write(f"Confusion Matrix:\n")
                f.write(f"True\\Pred  Good  Bad\n")
                f.write(f"Good      {cm[0,0]:4d}  {cm[0,1]:3d}\n")
                f.write(f"Bad       {cm[1,0]:4d}  {cm[1,1]:3d}\n")
            
            # Plot score histogram
            plt.figure(figsize=(10, 6))
            
            good_scores = [s for s, l in zip(scores, labels) if l == 0]
            bad_scores = [s for s, l in zip(scores, labels) if l == 1]
            
            plt.hist(good_scores, bins=30, alpha=0.7, label='Good Welds', color='green')
            plt.hist(bad_scores, bins=30, alpha=0.7, label='Bad Welds', color='red')
            plt.axvline(self.threshold, color='black', linestyle='--', label=f'Threshold: {self.threshold:.3f}')
            
            plt.xlabel('Anomaly Score')
            plt.ylabel('Frequency')
            plt.title('Distribution of Anomaly Scores')
            plt.legend()
            plt.savefig(save_path / 'score_distribution.png')
            plt.close()
            
            print(f"Results saved to: {save_path}")
        
        return auc_score, predictions, labels

def main():
    """Main pipeline for welding anomaly detection"""
    
    # Configuration
    base_dir = Path(__file__).parent.parent
    train_dir = base_dir / 'extracted_welds' / 'train'
    test_dir = base_dir / 'extracted_welds' / 'val'  # Use validation set as test
    results_dir = base_dir / 'patchcore_results'
    
    print("=== Welding Anomaly Detection with PatchCore ===")
    
    # Check if extracted patches exist
    if not train_dir.exists():
        print(f"Error: Training patches not found at {train_dir}")
        print("Please run extract_welds.py first to extract welding patches.")
        return
    
    # Initialize PatchCore
    patchcore = WeldingPatchCore(
        train_data_dir=str(train_dir),
        test_data_dir=str(test_dir)
    )
    
    # Build memory bank from good welding patches
    patchcore.build_memory_bank(subsample_ratio=0.1)
    
    # Compute threshold
    training_scores = patchcore.compute_threshold()
    
    # Evaluate on test data
    if test_dir.exists():
        auc_score, predictions, labels = patchcore.evaluate(
            test_data_dir=str(test_dir),
            save_dir=str(results_dir)
        )
        
        print(f"\nFinal AUC-ROC Score: {auc_score:.4f}")
    else:
        print(f"Warning: Test directory {test_dir} not found")

if __name__ == "__main__":
    main()
