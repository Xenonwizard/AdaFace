import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
from glob import glob
from face_alignment import align
from inference import load_pretrained_model, to_input
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import json

class YangMiDataset(Dataset):
    """Custom dataset for YangMi images"""
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load and align face
        try:
            aligned_rgb_img = align.get_aligned_face(img_path)
            if aligned_rgb_img is None:
                # Return a dummy tensor if face alignment fails
                return torch.zeros(3, 112, 112), label
            
            # Convert to model input format
            bgr_input = to_input(aligned_rgb_img)
            return bgr_input.squeeze(0), label  # Remove batch dimension
            
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return torch.zeros(3, 112, 112), label

class AdaFaceTransferLearner:
    def __init__(self, model_name='ir_50', num_classes=2):
        """
        Initialize AdaFace for transfer learning
        num_classes: 2 for binary (YangMi vs Not-YangMi), or more for multi-class
        """
        print(f"Loading AdaFace model: {model_name}")
        self.model = load_pretrained_model(model_name)
        self.num_classes = num_classes
        
        # Freeze early layers (backbone)
        self.freeze_backbone()
        
        # Modify final layer for your classification task
        self.modify_classifier()
        
        # Move to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        print(f"Using device: {self.device}")
        
    def freeze_backbone(self):
        """Freeze the backbone (early layers) of the model"""
        print("Freezing backbone layers...")
        
        # Freeze all parameters first
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Unfreeze only the final layers (usually the last few layers)
        # This depends on your model architecture - adjust as needed
        if hasattr(self.model, 'head'):
            for param in self.model.head.parameters():
                param.requires_grad = True
        elif hasattr(self.model, 'fc'):
            for param in self.model.fc.parameters():
                param.requires_grad = True
        elif hasattr(self.model, 'classifier'):
            for param in self.model.classifier.parameters():
                param.requires_grad = True
        
        # Count trainable parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
        
    def modify_classifier(self):
        """Modify the final classification layer"""
        # Add a custom classifier head
        # This assumes the model outputs features - adjust based on your model architecture
        
        # Get the feature dimension (usually 512 for most face models)
        feature_dim = 512  # Adjust this based on your model
        
        # Add a simple classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, self.num_classes)
        ).to(self.device)
        
        print(f"Added classifier for {self.num_classes} classes")
    
    def prepare_data(self, yangmi_dir, negative_dir=None, test_size=0.2):
        """
        Prepare training data
        yangmi_dir: Directory with YangMi images
        negative_dir: Directory with non-YangMi images (optional)
        """
        print("Preparing dataset...")
        
        image_paths = []
        labels = []
        
        # Get YangMi images (label = 1)
        yangmi_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        yangmi_files = []
        for ext in yangmi_extensions:
            yangmi_files.extend(glob(os.path.join(yangmi_dir, ext)))
            yangmi_files.extend(glob(os.path.join(yangmi_dir, ext.upper())))
        
        image_paths.extend(yangmi_files)
        labels.extend([1] * len(yangmi_files))
        
        # Get negative samples if provided (label = 0)
        if negative_dir and os.path.exists(negative_dir):
            neg_files = []
            for ext in yangmi_extensions:
                neg_files.extend(glob(os.path.join(negative_dir, ext)))
                neg_files.extend(glob(os.path.join(negative_dir, ext.upper())))
            
            image_paths.extend(neg_files)
            labels.extend([0] * len(neg_files))
        
        print(f"Total images: {len(image_paths)}")
        print(f"YangMi images: {labels.count(1)}")
        print(f"Negative images: {labels.count(0)}")
        
        # Split into train/validation
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            image_paths, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        # Create datasets
        train_dataset = YangMiDataset(train_paths, train_labels)
        val_dataset = YangMiDataset(val_paths, val_labels)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
        
        return train_loader, val_loader
    
    def train_model(self, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
        """Train the model using transfer learning"""
        print("Starting transfer learning training...")
        
        # Setup optimizer - only train unfrozen parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        trainable_params.extend(self.classifier.parameters())
        
        optimizer = optim.Adam(trainable_params, lr=learning_rate, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        best_val_acc = 0.0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # Training phase
            self.model.train()
            self.classifier.train()
            
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc="Training")):
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Skip batch if any images are invalid
                if torch.any(torch.sum(images.view(images.size(0), -1), dim=1) == 0):
                    continue
                
                optimizer.zero_grad()
                
                # Forward pass
                features, _ = self.model(images)  # Get features from AdaFace
                outputs = self.classifier(features)  # Classify using our custom head
                
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                # Statistics
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            # Validation phase
            self.model.eval()
            self.classifier.eval()
            
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in tqdm(val_loader, desc="Validation"):
                    images, labels = images.to(self.device), labels.to(self.device)
                    
                    # Skip batch if any images are invalid
                    if torch.any(torch.sum(images.view(images.size(0), -1), dim=1) == 0):
                        continue
                    
                    features, _ = self.model(images)
                    outputs = self.classifier(features)
                    
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            # Calculate metrics
            train_acc = 100 * train_correct / train_total if train_total > 0 else 0
            val_acc = 100 * val_correct / val_total if val_total > 0 else 0
            
            # Store history
            history['train_loss'].append(train_loss / len(train_loader))
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss / len(val_loader))
            history['val_acc'].append(val_acc)
            
            # Print epoch results
            print(f"Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model('best_yangmi_model.pth')
                print(f"New best model saved! Val Acc: {val_acc:.2f}%")
            
            scheduler.step()
        
        print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.2f}%")
        return history
    
    def save_model(self, filepath):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'classifier_state_dict': self.classifier.state_dict(),
            'num_classes': self.num_classes
        }, filepath)
    
    def load_model(self, filepath):
        """Load a trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
        self.num_classes = checkpoint['num_classes']
    
    def plot_training_history(self, history, save_path='training_history.png'):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        ax1.plot(history['train_loss'], label='Train Loss')
        ax1.plot(history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(history['train_acc'], label='Train Accuracy')
        ax2.plot(history['val_acc'], label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def evaluate_on_test(self, test_dir):
        """Evaluate the trained model on test images"""
        print("Evaluating on test images...")
        
        # Get test images
        test_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        test_files = []
        for ext in test_extensions:
            test_files.extend(glob(os.path.join(test_dir, ext)))
            test_files.extend(glob(os.path.join(test_dir, ext.upper())))
        
        self.model.eval()
        self.classifier.eval()
        
        results = []
        
        with torch.no_grad():
            for img_path in tqdm(test_files, desc="Testing"):
                try:
                    # Align face
                    aligned_rgb_img = align.get_aligned_face(img_path)
                    if aligned_rgb_img is None:
                        continue
                    
                    # Convert to model input
                    bgr_input = to_input(aligned_rgb_img).to(self.device)
                    
                    # Get prediction
                    features, _ = self.model(bgr_input)
                    outputs = self.classifier(features)
                    probabilities = torch.softmax(outputs, dim=1)
                    _, predicted = torch.max(outputs, 1)
                    
                    # Store result
                    yangmi_prob = probabilities[0][1].cpu().item()  # Probability of being YangMi
                    is_yangmi = predicted.cpu().item() == 1
                    
                    results.append({
                        'image': os.path.basename(img_path),
                        'path': img_path,
                        'is_yangmi': is_yangmi,
                        'yangmi_probability': yangmi_prob,
                        'confidence': max(probabilities[0]).cpu().item()
                    })
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
        
        return results

def main():
    """Main training function"""
    # Initialize transfer learner
    learner = AdaFaceTransferLearner(model_name='ir_50', num_classes=2)
    
    # Set your data paths
    yangmi_dir = 'elifiles/images/yangmi'  # Your YangMi training images
    # negative_dir = 'elifiles/images/others'  # Other people's images (optional)
    test_dir = 'elifiles/images/yangmi_test'  # Test images
    
    try:
        # Prepare data
        train_loader, val_loader = learner.prepare_data(yangmi_dir, negative_dir)
        
        # Train model
        history = learner.train_model(train_loader, val_loader, num_epochs=15, learning_rate=0.001)
        
        # Plot training history
        learner.plot_training_history(history)
        
        # Evaluate on test set
        test_results = learner.evaluate_on_test(test_dir)
        
        # Save results
        results_df = pd.DataFrame(test_results)
        results_df.to_csv('yangmi_transfer_learning_results.csv', index=False)
        
        # Print summary
        if len(test_results) > 0:
            yangmi_detected = sum(1 for r in test_results if r['is_yangmi'])
            avg_confidence = np.mean([r['confidence'] for r in test_results])
            
            print(f"\n=== TRANSFER LEARNING RESULTS ===")
            print(f"Test images processed: {len(test_results)}")
            print(f"Images classified as YangMi: {yangmi_detected}")
            print(f"Average confidence: {avg_confidence:.3f}")
            print(f"Results saved to: yangmi_transfer_learning_results.csv")
        
    except Exception as e:
        print(f"Error during training: {e}")
        print("\nMake sure you have:")
        print("1. YangMi images in the specified directory")
        print("2. Optionally, negative samples in another directory")
        print("3. All dependencies installed")

if __name__ == "__main__":
    main()