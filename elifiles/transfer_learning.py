import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
import cv2
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import json
import warnings
warnings.filterwarnings('ignore')

# AdaFace specific imports (you'll need these files from the AdaFace repo)
import sys
sys.path.append('.')  # Assuming AdaFace files are in current directory

# You'll need to download these from https://github.com/mk-minchul/AdaFace
try:
    import net  # AdaFace network architecture
    from head import AdaFace as AdaFaceHead  # AdaFace loss head
    from face_alignment import align  # MTCNN face alignment
except ImportError:
    print("ERROR: AdaFace modules not found!")
    print("Please download the following files from https://github.com/mk-minchul/AdaFace:")
    print("- net.py")
    print("- head.py") 
    print("- face_alignment/")
    print("- pretrained/ (with adaface_ir50_ms1mv2.ckpt)")
    exit(1)

class AdaFaceYangMiDataset(Dataset):
    """Dataset for AdaFace fine-tuning with proper MTCNN alignment"""
    
    def __init__(self, image_paths, labels, is_training=True):
        self.image_paths = image_paths
        self.labels = labels
        self.is_training = is_training
        
        # AdaFace specific preprocessing (BGR, 112x112, mean=0.5, std=0.5)
        if is_training:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
            ])
    
    def to_adaface_input(self, pil_rgb_image):
        """Convert PIL RGB image to AdaFace input format (BGR, normalized)"""
        # Convert PIL to numpy
        np_img = np.array(pil_rgb_image)
        # Convert RGB to BGR and normalize to [-1, 1]
        bgr_img = ((np_img[:,:,::-1] / 255.) - 0.5) / 0.5
        # Convert to tensor and add batch dimension
        tensor = torch.tensor([bgr_img.transpose(2,0,1)]).float()
        return tensor.squeeze(0)  # Remove batch dimension for dataset
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            # Use AdaFace's MTCNN alignment (much better than OpenCV)
            try:
                aligned_rgb_img = align.get_aligned_face(img_path)
                if aligned_rgb_img is None:
                    raise Exception("MTCNN failed")
            except:
                # Fallback: manual preprocessing
                image = cv2.imread(img_path)
                if image is None:
                    return torch.zeros(3, 112, 112), label
                
                # Convert BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # Resize to 112x112 (AdaFace standard)
                aligned_rgb_img = cv2.resize(image, (112, 112))
                # Convert to PIL
                from PIL import Image
                aligned_rgb_img = Image.fromarray(aligned_rgb_img)
            
            # Apply transforms if training
            if self.is_training:
                aligned_rgb_img = self.transform(aligned_rgb_img)
                # Convert tensor back to PIL for AdaFace processing
                if isinstance(aligned_rgb_img, torch.Tensor):
                    aligned_rgb_img = transforms.ToPILImage()(aligned_rgb_img)
            
            # Convert to AdaFace input format
            adaface_input = self.to_adaface_input(aligned_rgb_img)
            
            return adaface_input, label
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            return torch.zeros(3, 112, 112), label

class AdaFaceYangMiFinetuner:
    """Fine-tune AdaFace pretrained model for YangMi recognition"""
    
    def __init__(self, architecture='ir_50', pretrained_path='pretrained/adaface_ir50_ms1mv2.ckpt'):
        print(f"Initializing AdaFace fine-tuner with {architecture}")
        self.architecture = architecture
        self.pretrained_path = pretrained_path
        
        # Load pretrained AdaFace model
        self.load_pretrained_model()
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Move model to device
        self.backbone.to(self.device)
        
        # We'll create a new classification head for YangMi vs Others
        self.setup_classification_head()
        
        print("AdaFace fine-tuner initialized successfully!")
    
    def load_pretrained_model(self):
        """Load pretrained AdaFace backbone"""
        print(f"Loading pretrained AdaFace model from {self.pretrained_path}")
        
        if not os.path.exists(self.pretrained_path):
            raise FileNotFoundError(f"Pretrained model not found: {self.pretrained_path}")
        
        # Build AdaFace network
        self.backbone = net.build_model(self.architecture)
        
        # Load pretrained weights
        checkpoint = torch.load(self.pretrained_path, map_location='cpu')
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            # Remove 'model.' prefix if present
            model_state_dict = {key.replace('model.', ''): val for key, val in state_dict.items() if key.startswith('model.')}
        else:
            model_state_dict = checkpoint
        
        # Load weights
        self.backbone.load_state_dict(model_state_dict, strict=False)
        print("✓ Pretrained AdaFace weights loaded successfully")
    
    def setup_classification_head(self):
        """Setup classification head for YangMi recognition"""
        # Get feature dimension from backbone
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 112, 112)
            if self.device.type == 'cuda':
                dummy_input = dummy_input.to(self.device)
                features, _ = self.backbone(dummy_input)
            else:
                features, _ = self.backbone(dummy_input)
            feature_dim = features.shape[1]
        
        print(f"AdaFace feature dimension: {feature_dim}")
        
        # Create classification head for binary classification
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256), 
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2)  # YangMi vs Others
        ).to(self.device)
        
        print("✓ Classification head created")
    
    def freeze_backbone_layers(self, freeze_ratio=0.7):
        """Freeze early layers of the backbone for fine-tuning"""
        print(f"Freezing {freeze_ratio*100}% of backbone layers...")
        
        # Get all backbone parameters
        backbone_params = list(self.backbone.named_parameters())
        num_layers = len(backbone_params)
        freeze_until = int(num_layers * freeze_ratio)
        
        # Freeze early layers
        for i, (name, param) in enumerate(backbone_params):
            if i < freeze_until:
                param.requires_grad = False
            else:
                param.requires_grad = True
        
        # Count trainable parameters
        backbone_trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        classifier_trainable = sum(p.numel() for p in self.classifier.parameters())
        total_trainable = backbone_trainable + classifier_trainable
        
        print(f"Trainable parameters:")
        print(f"  Backbone: {backbone_trainable:,}")
        print(f"  Classifier: {classifier_trainable:,}")
        print(f"  Total: {total_trainable:,}")
    
    def prepare_training_data(self, gallery_dir, negative_dir=None, test_size=0.2, batch_size=16):
        """Prepare training data for AdaFace fine-tuning"""
        print("Preparing AdaFace training dataset...")
        
        # Get positive samples (YangMi)
        positive_files = self._get_image_files(gallery_dir)
        print(f"Found {len(positive_files)} YangMi images")
        
        # Get negative samples
        if negative_dir and os.path.exists(negative_dir):
            negative_files = self._get_image_files(negative_dir)
            print(f"Found {len(negative_files)} negative images")
        else:
            print("No negative directory provided")
            negative_files = []
        
        # Create balanced dataset
        if len(negative_files) == 0:
            # Use some positives as negatives with heavy augmentation
            negative_files = positive_files[:len(positive_files)//2]
            print(f"Using {len(negative_files)} augmented positives as negatives")
        
        # Balance the dataset
        min_samples = min(len(positive_files), len(negative_files))
        positive_files = positive_files[:min_samples]
        negative_files = negative_files[:min_samples]
        
        # Combine data
        all_files = positive_files + negative_files
        all_labels = [1] * len(positive_files) + [0] * len(negative_files)
        
        print(f"Balanced dataset:")
        print(f"  Positive samples: {len(positive_files)}")
        print(f"  Negative samples: {len(negative_files)}")
        print(f"  Total samples: {len(all_files)}")
        
        # Split into train/validation
        train_files, val_files, train_labels, val_labels = train_test_split(
            all_files, all_labels, 
            test_size=test_size, 
            random_state=42, 
            stratify=all_labels
        )
        
        # Create datasets
        train_dataset = AdaFaceYangMiDataset(train_files, train_labels, is_training=True)
        val_dataset = AdaFaceYangMiDataset(val_files, val_labels, is_training=False)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=2,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=2,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        return train_loader, val_loader
    
    def _get_image_files(self, directory):
        """Get all image files from directory"""
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob(os.path.join(directory, ext)))
            image_files.extend(glob(os.path.join(directory, ext.upper())))
        return image_files
    
    def train_model(self, train_loader, val_loader, num_epochs=20, learning_rate=0.0001):
        """Fine-tune AdaFace model"""
        print("Starting AdaFace fine-tuning...")
        
        # Freeze backbone layers
        self.freeze_backbone_layers(freeze_ratio=0.8)
        
        # Setup optimizer with different learning rates
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        classifier_params = list(self.classifier.parameters())
        
        optimizer = optim.Adam([
            {'params': backbone_params, 'lr': learning_rate * 0.1},  # Lower LR for backbone
            {'params': classifier_params, 'lr': learning_rate}       # Higher LR for classifier
        ], weight_decay=1e-4)
        
        # Loss function and scheduler
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)
        
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
            self.backbone.train()
            self.classifier.train()
            
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for images, labels in tqdm(train_loader, desc="Training"):
                try:
                    images, labels = images.to(self.device), labels.to(self.device)
                    
                    # Skip problematic batches
                    if torch.any(torch.isnan(images)) or images.size(0) == 0:
                        continue
                    
                    optimizer.zero_grad()
                    
                    # Forward pass through AdaFace backbone
                    features, norms = self.backbone(images)
                    
                    # Forward pass through classifier
                    outputs = self.classifier(features)
                    
                    # Compute loss
                    loss = criterion(outputs, labels)
                    
                    # Backward pass
                    loss.backward()
                    
                    # Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(
                        list(self.backbone.parameters()) + list(self.classifier.parameters()), 
                        max_norm=1.0
                    )
                    
                    optimizer.step()
                    
                    # Statistics
                    train_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    train_total += labels.size(0)
                    train_correct += (predicted == labels).sum().item()
                    
                except Exception as e:
                    if "out of memory" in str(e):
                        print("GPU memory issue, skipping batch")
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                        continue
                    else:
                        print(f"Training error: {e}")
                        continue
            
            # Validation phase
            self.backbone.eval()
            self.classifier.eval()
            
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in tqdm(val_loader, desc="Validation"):
                    try:
                        images, labels = images.to(self.device), labels.to(self.device)
                        
                        if torch.any(torch.isnan(images)) or images.size(0) == 0:
                            continue
                        
                        # Forward pass
                        features, norms = self.backbone(images)
                        outputs = self.classifier(features)
                        
                        loss = criterion(outputs, labels)
                        
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()
                        
                    except Exception as e:
                        print(f"Validation error: {e}")
                        continue
            
            # Calculate metrics
            train_acc = 100 * train_correct / train_total if train_total > 0 else 0
            val_acc = 100 * val_correct / val_total if val_total > 0 else 0
            
            # Store history
            train_loss_avg = train_loss / max(len(train_loader), 1)
            val_loss_avg = val_loss / max(len(val_loader), 1)
            
            history['train_loss'].append(train_loss_avg)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss_avg)
            history['val_acc'].append(val_acc)
            
            # Print results
            print(f"Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model('best_adaface_yangmi_model.pth')
                print(f"✓ New best model saved! Val Acc: {val_acc:.2f}%")
            
            scheduler.step()
        
        print(f"\nFine-tuning completed! Best validation accuracy: {best_val_acc:.2f}%")
        return history
    
    def extract_features(self, image_path):
        """Extract AdaFace features from image"""
        try:
            # Use AdaFace preprocessing
            try:
                aligned_rgb_img = align.get_aligned_face(image_path)
                if aligned_rgb_img is None:
                    raise Exception("MTCNN failed")
            except:
                # Fallback preprocessing
                image = cv2.imread(image_path)
                if image is None:
                    return None
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                aligned_rgb_img = cv2.resize(image, (112, 112))
                from PIL import Image
                aligned_rgb_img = Image.fromarray(aligned_rgb_img)
            
            # Convert to AdaFace input
            dataset = AdaFaceYangMiDataset([image_path], [0], is_training=False)
            adaface_input, _ = dataset[0]
            
            # Extract features
            self.backbone.eval()
            with torch.no_grad():
                adaface_input = adaface_input.unsqueeze(0).to(self.device)
                features, norms = self.backbone(adaface_input)
                
                # Normalize features
                features = features / torch.norm(features, dim=1, keepdim=True)
                
            return features.cpu().numpy().flatten()
            
        except Exception as e:
            print(f"Error extracting features from {image_path}: {e}")
            return None
    
    def evaluate_on_test(self, test_dir, threshold=0.5):
        """Evaluate fine-tuned AdaFace model on test images"""
        print(f"Evaluating fine-tuned AdaFace on test images from {test_dir}")
        
        # Get test images
        test_files = self._get_image_files(test_dir)
        
        if len(test_files) == 0:
            print(f"No test images found in {test_dir}")
            return []
        
        print(f"Found {len(test_files)} test images")
        
        # Evaluate each image
        results = []
        self.backbone.eval()
        self.classifier.eval()
        
        with torch.no_grad():
            for img_path in tqdm(test_files, desc="Processing test images"):
                try:
                    # Load and process image
                    temp_dataset = AdaFaceYangMiDataset([img_path], [0], is_training=False)
                    adaface_input, _ = temp_dataset[0]
                    
                    if torch.sum(adaface_input) == 0:
                        continue
                    
                    adaface_input = adaface_input.unsqueeze(0).to(self.device)
                    
                    # Get AdaFace features
                    features, norms = self.backbone(adaface_input)
                    
                    # Get classification prediction
                    outputs = self.classifier(features)
                    probabilities = torch.softmax(outputs, dim=1)
                    _, predicted = torch.max(outputs, 1)
                    
                    # Store result
                    yangmi_prob = probabilities[0][1].cpu().item()
                    is_yangmi = predicted.cpu().item() == 1
                    confidence = max(probabilities[0]).cpu().item()
                    feature_norm = norms[0].cpu().item()
                    
                    results.append({
                        'test_image': os.path.basename(img_path),
                        'yangmi_probability': yangmi_prob,
                        'is_yangmi': is_yangmi,
                        'confidence': confidence,
                        'feature_norm': feature_norm,  # AdaFace quality indicator
                        'similarity': yangmi_prob,
                        'is_match': yangmi_prob > threshold,
                        'test_path': img_path
                    })
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    continue
        
        return results
    
    def save_model(self, filepath):
        """Save fine-tuned AdaFace model"""
        try:
            torch.save({
                'backbone_state_dict': self.backbone.state_dict(),
                'classifier_state_dict': self.classifier.state_dict(),
                'architecture': self.architecture
            }, filepath)
            print(f"Model saved to {filepath}")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def load_model(self, filepath):
        """Load fine-tuned AdaFace model"""
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            self.backbone.load_state_dict(checkpoint['backbone_state_dict'])
            self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
            print(f"Fine-tuned model loaded from {filepath}")
        except Exception as e:
            print(f"Error loading model: {e}")

def main():
    """Main function for AdaFace fine-tuning"""
    print("=== AdaFace Fine-tuning for YangMi Recognition ===")
    
    # Initialize fine-tuner
    finetuner = AdaFaceYangMiFinetuner(
        architecture='ir_50',
        pretrained_path='pretrained/adaface_ir50_ms1mv2.ckpt'
    )
    
    # Set paths
    gallery_dir = 'elifiles/images/yangmi'        # YangMi training images
    negative_dir = 'elifiles/images/others'       # Other people's images
    test_dir = 'elifiles/images/yangmi_test'      # Test images
    
    try:
        print("\n=== Preparing Training Data ===")
        train_loader, val_loader = finetuner.prepare_training_data(
            gallery_dir, negative_dir, batch_size=8
        )
        
        print("\n=== Fine-tuning AdaFace Model ===")
        history = finetuner.train_model(train_loader, val_loader, num_epochs=15)
        
        # Test the fine-tuned model
        if os.path.exists(test_dir):
            print("\n=== Evaluating Fine-tuned AdaFace Model ===")
            results = finetuner.evaluate_on_test(test_dir)
            
            if results:
                print(f"Test evaluation completed: {len(results)} images processed")
                
                # Create results report
                df = pd.DataFrame(results)
                df.to_csv('adaface_finetuning_results.csv', index=False)
                
                # Summary statistics
                yangmi_detected = sum(1 for r in results if r['is_yangmi'])
                avg_prob = np.mean([r['yangmi_probability'] for r in results])
                avg_conf = np.mean([r['confidence'] for r in results])
                avg_norm = np.mean([r['feature_norm'] for r in results])
                
                print(f"\n=== FINAL RESULTS SUMMARY ===")
                print(f"Images classified as YangMi: {yangmi_detected}/{len(results)}")
                print(f"Average YangMi probability: {avg_prob:.4f}")
                print(f"Average confidence: {avg_conf:.4f}")
                print(f"Average feature norm: {avg_norm:.4f}")
                print(f"Results saved to 'adaface_finetuning_results.csv'")
        
        print("\n=== Fine-tuning Complete ===")
        print("Check 'best_adaface_yangmi_model.pth' for the trained model")
        
    except Exception as e:
        print(f"Error during fine-tuning: {e}")

if __name__ == "__main__":
    main()