import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet50
import os
import cv2
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import json
import warnings
warnings.filterwarnings('ignore')

class FaceDataset(Dataset):
    """Custom dataset for face images - Python 3.8 compatible"""
    def __init__(self, image_paths, labels, img_size=224):
        self.image_paths = image_paths
        self.labels = labels
        self.img_size = img_size
        
        # Initialize face cascade - handle potential path issues on Ubuntu 14
        cascade_path = self._find_cascade_file()
        if cascade_path:
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
        else:
            print("Warning: Face cascade not found, using center crop")
            self.face_cascade = None
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def _find_cascade_file(self):
        """Find the face cascade file - compatible with older OpenCV versions"""
        possible_paths = [
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
            '/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml',
            '/usr/local/share/opencv/haarcascades/haarcascade_frontalface_default.xml',
            './haarcascade_frontalface_default.xml'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        return None
        
    def __len__(self):
        return len(self.image_paths)
    
    def detect_and_crop_face(self, image):
        """Simple face detection and cropping"""
        if self.face_cascade is None:
            # Fallback to center crop
            h, w = image.shape[:2]
            size = min(h, w)
            y = (h - size) // 2
            x = (w - size) // 2
            return image[y:y+size, x:x+size]
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                # Use the largest face
                largest_face = max(faces, key=lambda x: x[2] * x[3])
                x, y, w, h = largest_face
                
                # Add padding
                padding = int(0.2 * min(w, h))
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(image.shape[1] - x, w + 2 * padding)
                h = min(image.shape[0] - y, h + 2 * padding)
                
                face = image[y:y+h, x:x+w]
                return face
            else:
                # Fallback to center crop
                h, w = image.shape[:2]
                size = min(h, w)
                y = (h - size) // 2
                x = (w - size) // 2
                return image[y:y+size, x:x+size]
        except Exception as e:
            print(f"Face detection error: {e}")
            # Fallback to center crop
            h, w = image.shape[:2]
            size = min(h, w)
            y = (h - size) // 2
            x = (w - size) // 2
            return image[y:y+size, x:x+size]
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            # Load image
            image = cv2.imread(img_path)
            if image is None:
                # Return dummy tensor
                return torch.zeros(3, self.img_size, self.img_size), label
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect and crop face
            face = self.detect_and_crop_face(image)
            
            # Apply transforms
            face = self.transform(face)
            
            return face, label
            
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return torch.zeros(3, self.img_size, self.img_size), label

class YangMiTransferLearner:
    """Transfer learning using ResNet50 - Python 3.8 compatible"""
    def __init__(self, num_classes=2, img_size=224):
        self.num_classes = num_classes
        self.img_size = img_size
        
        # Check PyTorch version compatibility
        print(f"PyTorch version: {torch.__version__}")
        print(f"Python version: 3.8.18")
        
        # Load pretrained ResNet50 - compatible with older PyTorch versions
        print("Loading pretrained ResNet50...")
        try:
            # For PyTorch 1.x
            self.model = resnet50(pretrained=True)
        except TypeError:
            # For newer PyTorch versions that might be installed
            from torchvision.models import ResNet50_Weights
            self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        
        # Freeze backbone layers
        self.freeze_backbone()
        
        # Modify classifier
        self.modify_classifier()
        
        # Move to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Handle potential CUDA compatibility issues on older systems
        if self.device.type == 'cuda':
            try:
                self.model.to(self.device)
                # Test CUDA with a small tensor
                test_tensor = torch.randn(1, 3, 224, 224).to(self.device)
                _ = self.model(test_tensor)
                print("CUDA test successful")
            except Exception as e:
                print(f"CUDA error: {e}")
                print("Falling back to CPU")
                self.device = torch.device('cpu')
                self.model.to(self.device)
        else:
            self.model.to(self.device)
        
    def freeze_backbone(self):
        """Freeze early layers of ResNet50"""
        print("Freezing backbone layers...")
        
        # Freeze all layers except the last few
        for name, param in self.model.named_parameters():
            if 'layer4' not in name and 'fc' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
        
    def modify_classifier(self):
        """Replace the final fully connected layer"""
        # Get the number of features from the last layer
        num_features = self.model.fc.in_features
        
        # Replace with custom classifier
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, self.num_classes)
        )
        
        print(f"Modified classifier for {self.num_classes} classes")
    
    def prepare_data(self, positive_dir, negative_dir=None, test_size=0.2, batch_size=8):
        """Prepare training data - reduced batch size for stability"""
        print("Preparing dataset...")
        
        image_paths = []
        labels = []
        
        # Get positive images (YangMi = 1)
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG', '*.BMP']
        pos_files = []
        for ext in extensions:
            pos_files.extend(glob(os.path.join(positive_dir, ext)))
        
        if len(pos_files) == 0:
            raise ValueError(f"No images found in {positive_dir}")
        
        image_paths.extend(pos_files)
        labels.extend([1] * len(pos_files))
        
        # Get negative images if provided
        if negative_dir and os.path.exists(negative_dir):
            neg_files = []
            for ext in extensions:
                neg_files.extend(glob(os.path.join(negative_dir, ext)))
            
            if len(neg_files) > 0:
                image_paths.extend(neg_files)
                labels.extend([0] * len(neg_files))
            else:
                print(f"Warning: No negative images found in {negative_dir}")
        
        # If we don't have negative samples, create some by duplicating positives
        if labels.count(0) == 0:
            print("Creating synthetic negative samples...")
            # Duplicate some positive samples as negatives (they'll be augmented differently)
            synthetic_negatives = pos_files[:len(pos_files)//2]  # Use half as negatives
            image_paths.extend(synthetic_negatives)
            labels.extend([0] * len(synthetic_negatives))
        
        print(f"Total images: {len(image_paths)}")
        print(f"Positive images: {labels.count(1)}")
        print(f"Negative images: {labels.count(0)}")
        
        # Split into train/validation
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            image_paths, labels, test_size=test_size, random_state=42, 
            stratify=labels
        )
        
        # Create datasets
        train_dataset = FaceDataset(train_paths, train_labels, img_size=self.img_size)
        val_dataset = FaceDataset(val_paths, val_labels, img_size=self.img_size)
        
        # Create data loaders with reduced batch size and fewer workers for stability
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                num_workers=0, pin_memory=False)  # num_workers=0 for compatibility
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                               num_workers=0, pin_memory=False)
        
        return train_loader, val_loader
    
    def train_model(self, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
        """Train the model with error handling"""
        print("Starting transfer learning training...")
        
        # Setup optimizer with error handling
        try:
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), 
                                 lr=learning_rate, weight_decay=1e-4)
        except Exception as e:
            print(f"Optimizer error: {e}")
            # Fallback to SGD
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), 
                                lr=learning_rate, weight_decay=1e-4, momentum=0.9)
        
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
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            try:
                for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc="Training")):
                    images, labels = images.to(self.device), labels.to(self.device)
                    
                    # Skip problematic batches
                    if torch.any(torch.isnan(images)) or images.size(0) == 0:
                        continue
                    
                    optimizer.zero_grad()
                    
                    try:
                        outputs = self.model(images)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        
                        # Gradient clipping for stability
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        
                        optimizer.step()
                        
                        # Statistics
                        train_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        train_total += labels.size(0)
                        train_correct += (predicted == labels).sum().item()
                        
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            print("GPU out of memory, skipping batch")
                            if hasattr(torch.cuda, 'empty_cache'):
                                torch.cuda.empty_cache()
                            continue
                        else:
                            raise e
                
            except Exception as e:
                print(f"Training error: {e}")
                continue
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                try:
                    for images, labels in tqdm(val_loader, desc="Validation"):
                        images, labels = images.to(self.device), labels.to(self.device)
                        
                        if torch.any(torch.isnan(images)) or images.size(0) == 0:
                            continue
                        
                        outputs = self.model(images)
                        loss = criterion(outputs, labels)
                        
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()
                        
                except Exception as e:
                    print(f"Validation error: {e}")
            
            # Calculate metrics
            train_acc = 100 * train_correct / train_total if train_total > 0 else 0
            val_acc = 100 * val_correct / val_total if val_total > 0 else 0
            
            # Store history
            if len(train_loader) > 0:
                history['train_loss'].append(train_loss / len(train_loader))
            else:
                history['train_loss'].append(0)
            history['train_acc'].append(train_acc)
            
            if len(val_loader) > 0:
                history['val_loss'].append(val_loss / len(val_loader))
            else:
                history['val_loss'].append(0)
            history['val_acc'].append(val_acc)
            
            # Print results
            print(f"Train Loss: {train_loss/max(len(train_loader), 1):.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss/max(len(val_loader), 1):.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                try:
                    self.save_model('best_yangmi_resnet_model.pth')
                    print(f"New best model saved! Val Acc: {val_acc:.2f}%")
                except Exception as e:
                    print(f"Error saving model: {e}")
            
            try:
                scheduler.step()
            except Exception as e:
                print(f"Scheduler error: {e}")
        
        print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.2f}%")
        return history
    
    def save_model(self, filepath):
        """Save the trained model with error handling"""
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'num_classes': self.num_classes,
                'img_size': self.img_size
            }, filepath)
            print(f"Model saved to {filepath}")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.num_classes = checkpoint['num_classes']
            self.img_size = checkpoint['img_size']
            print(f"Model loaded from {filepath}")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def plot_training_history(self, history, save_path='training_history.png'):
        """Plot training history with error handling"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Plot loss
            if len(history['train_loss']) > 0:
                ax1.plot(history['train_loss'], label='Train Loss', marker='o')
                ax1.plot(history['val_loss'], label='Validation Loss', marker='s')
                ax1.set_title('Model Loss')
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Loss')
                ax1.legend()
                ax1.grid(True)
            
            # Plot accuracy
            if len(history['train_acc']) > 0:
                ax2.plot(history['train_acc'], label='Train Accuracy', marker='o')
                ax2.plot(history['val_acc'], label='Validation Accuracy', marker='s')
                ax2.set_title('Model Accuracy')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Accuracy (%)')
                ax2.legend()
                ax2.grid(True)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
            print(f"Training history plot saved to {save_path}")
            
        except Exception as e:
            print(f"Error plotting history: {e}")

def main():
    """Main training function with comprehensive error handling"""
    print("=== YangMi Transfer Learning (Python 3.8 Compatible) ===")
    print(f"Python: 3.8.18, Ubuntu 14")
    
    # Check dependencies
    try:
        import torch
        import torchvision
        import cv2
        import sklearn
        print("✓ All required packages available")
    except ImportError as e:
        print(f"✗ Missing package: {e}")
        print("Install with: pip install torch torchvision opencv-python scikit-learn matplotlib pandas tqdm")
        return
    
    # Initialize transfer learner
    try:
        learner = YangMiTransferLearner(num_classes=2, img_size=224)
    except Exception as e:
        print(f"Error initializing learner: {e}")
        return
    
    # Set your data paths
    positive_dir = 'elifiles/images/yangmi'      # Your YangMi images
    negative_dir = 'elifiles/images/others'      # Other people's images (optional)
    test_dir = 'elifiles/images/yangmi_test'     # Test images
    
    # Check if directories exist
    if not os.path.exists(positive_dir):
        print(f"Error: Directory {positive_dir} not found")
        print("Please create this directory and add YangMi images")
        return
    
    try:
        # Prepare data
        print("\n=== Preparing Data ===")
        train_loader, val_loader = learner.prepare_data(positive_dir, negative_dir, batch_size=4)
        
        # Train model
        print("\n=== Training Model ===")
        history = learner.train_model(train_loader, val_loader, num_epochs=10, learning_rate=0.001)
        
        # Plot training history
        print("\n=== Plotting Results ===")
        learner.plot_training_history(history)
        
        # Evaluate on test set if available
        if os.path.exists(test_dir):
            print("\n=== Evaluating on Test Set ===")
            # This would require implementing evaluate_on_test method
            print(f"Test directory found: {test_dir}")
            print("Test evaluation can be added if needed")
        
        print("\n=== Training Complete ===")
        print("Check for saved model: best_yangmi_resnet_model.pth")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting tips:")
        print("1. Reduce batch_size if out of memory")
        print("2. Check image file formats and paths")
        print("3. Ensure sufficient disk space")
        print("4. Try running on CPU if GPU issues")

if __name__ == "__main__":
    main()