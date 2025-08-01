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

# AdaFace specific imports
try:
    import net  # AdaFace network architecture
    from head import AdaFace as AdaFaceHead  # AdaFace loss head
    print("✓ AdaFace core modules imported successfully")
except ImportError as e:
    print(f"ERROR importing AdaFace modules: {e}")
    exit(1)

# Try to import face alignment
try:
    from face_alignment import align  # MTCNN face alignment
    MTCNN_AVAILABLE = True
    print("✓ MTCNN face alignment available")
except ImportError as e:
    print(f"Warning: MTCNN not available ({e}), will use fallback alignment")
    MTCNN_AVAILABLE = False

class AdaFaceMultiEthnicDataset(Dataset):
    """Dataset for AdaFace training with multi-ethnic negative samples"""
    
    def __init__(self, image_paths, labels, ethnic_groups, is_training=True):
        self.image_paths = image_paths
        self.labels = labels
        self.ethnic_groups = ethnic_groups
        self.is_training = is_training
        
        # We'll handle transforms manually in __getitem__ to avoid PIL/Tensor conflicts
        print(f"Dataset initialized with {len(image_paths)} images, training mode: {is_training}")
    
    def to_adaface_input(self, pil_rgb_image):
        """Convert PIL RGB image to AdaFace input format (BGR, normalized)"""
        np_img = np.array(pil_rgb_image)
        bgr_img = ((np_img[:,:,::-1] / 255.) - 0.5) / 0.5
        tensor = torch.tensor(bgr_img.transpose(2,0,1)).float()
        return tensor
    
    def fallback_alignment(self, image_path):
        """Fallback face alignment when MTCNN is not available"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Simple center crop and resize to 112x112
            h, w = image.shape[:2]
            size = min(h, w)
            y = (h - size) // 2
            x = (w - size) // 2
            cropped = image[y:y+size, x:x+size]
            
            # Resize to AdaFace standard size
            aligned = cv2.resize(cropped, (112, 112))
            
            from PIL import Image
            return Image.fromarray(aligned)
            
        except Exception as e:
            print(f"Fallback alignment error for {image_path}: {e}")
            return None
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        ethnic_group = self.ethnic_groups[idx]
        
        try:
            # Try MTCNN alignment first
            if MTCNN_AVAILABLE:
                try:
                    aligned_rgb_img = align.get_aligned_face(img_path)
                    if aligned_rgb_img is None:
                        raise Exception("MTCNN failed")
                except:
                    aligned_rgb_img = self.fallback_alignment(img_path)
            else:
                aligned_rgb_img = self.fallback_alignment(img_path)
            
            if aligned_rgb_img is None:
                return torch.zeros(3, 112, 112), label, ethnic_group
            
            # Convert to tensor first, then apply transforms
            if not isinstance(aligned_rgb_img, torch.Tensor):
                # Convert PIL to tensor
                aligned_tensor = transforms.ToTensor()(aligned_rgb_img)
            else:
                aligned_tensor = aligned_rgb_img
            
            # Apply training transforms if needed
            if self.is_training:
                # Apply transforms that work on tensors
                if torch.rand(1) > 0.5:  # Random horizontal flip
                    aligned_tensor = torch.flip(aligned_tensor, [2])
                
                # Color jitter simulation (simple brightness/contrast)
                if torch.rand(1) > 0.5:
                    brightness_factor = 0.8 + torch.rand(1) * 0.4  # 0.8 to 1.2
                    aligned_tensor = aligned_tensor * brightness_factor
                    aligned_tensor = torch.clamp(aligned_tensor, 0, 1)
            
            # Convert tensor back to PIL for AdaFace input conversion
            aligned_pil = transforms.ToPILImage()(aligned_tensor)
            
            # Convert to AdaFace input format
            adaface_input = self.to_adaface_input(aligned_pil)
            
            return adaface_input, label, ethnic_group
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            return torch.zeros(3, 112, 112), label, ethnic_group

class AdaFaceYangMiMultiEthnicTrainer:
    """Train AdaFace for YangMi recognition with multi-ethnic negative samples"""
    
    def __init__(self, architecture='ir_50', pretrained_path='pretrained/adaface_ir50_ms1mv2.ckpt'):
        print(f"Initializing AdaFace multi-ethnic trainer with {architecture}")
        self.architecture = architecture
        self.pretrained_path = pretrained_path
        
        # Load pretrained AdaFace model
        self.load_pretrained_model()
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Move model to device
        self.backbone.to(self.device)
        
        # Setup classification head
        self.setup_classification_head()
        
        print("AdaFace multi-ethnic trainer initialized successfully!")
    
    def load_pretrained_model(self):
        """Load pretrained AdaFace backbone"""
        print(f"Loading pretrained AdaFace model from {self.pretrained_path}")
        
        if not os.path.exists(self.pretrained_path):
            print(f"Available models in pretrained/:")
            for f in os.listdir('pretrained/'):
                print(f"  - {f}")
            raise FileNotFoundError(f"Pretrained model not found: {self.pretrained_path}")
        
        # Build AdaFace network
        self.backbone = net.build_model(self.architecture)
        
        # Load pretrained weights
        checkpoint = torch.load(self.pretrained_path, map_location='cpu')
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            model_state_dict = {key.replace('model.', '') if key.startswith('model.') else key: val 
                              for key, val in state_dict.items() if 'model.' in key or not key.startswith('head.')}
        else:
            model_state_dict = checkpoint
        
        # Load weights (ignore missing keys from head)
        missing_keys, unexpected_keys = self.backbone.load_state_dict(model_state_dict, strict=False)
        print(f"✓ Pretrained AdaFace weights loaded (missing: {len(missing_keys)}, unexpected: {len(unexpected_keys)})")
    
    def setup_classification_head(self):
        """Setup classification head for YangMi vs Others"""
        # Get feature dimension from backbone
        self.backbone.eval()  # Set to eval mode to avoid BatchNorm issues
        with torch.no_grad():
            dummy_input = torch.randn(2, 3, 112, 112)  # Use batch size of 2 instead of 1
            if self.device.type == 'cuda':
                dummy_input = dummy_input.to(self.device)
            
            try:
                features, _ = self.backbone(dummy_input)
                feature_dim = features.shape[1]
            except:
                backbone_output = self.backbone(dummy_input)
                if isinstance(backbone_output, tuple):
                    features = backbone_output[0]
                else:
                    features = backbone_output
                feature_dim = features.shape[1]
        
        print(f"AdaFace feature dimension: {feature_dim}")
        
        # Create classification head for binary classification (YangMi vs Others)
        # Use simpler architecture to avoid BatchNorm issues with small batches
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2)  # YangMi (1) vs Others (0)
        ).to(self.device)
        
        print("✓ Classification head created for YangMi vs Others")
    
    def prepare_multiethnic_data(self, yangmi_dir, ethnic_dirs, test_size=0.2, batch_size=16):
        """Prepare training data with YangMi and multi-ethnic negative samples"""
        print("Preparing multi-ethnic training dataset...")
        
        all_files = []
        all_labels = []
        all_ethnic_groups = []
        
        # Get YangMi images (positive samples)
        yangmi_files = self._get_image_files(yangmi_dir)
        print(f"Found {len(yangmi_files)} YangMi images")
        
        all_files.extend(yangmi_files)
        all_labels.extend([1] * len(yangmi_files))  # Label 1 for YangMi
        all_ethnic_groups.extend(['yangmi'] * len(yangmi_files))
        
        # Get negative samples from each ethnic group
        total_negatives = 0
        for ethnic_name, ethnic_dir in ethnic_dirs.items():
            if os.path.exists(ethnic_dir):
                ethnic_files = self._get_image_files(ethnic_dir)
                print(f"Found {len(ethnic_files)} {ethnic_name} images")
                
                all_files.extend(ethnic_files)
                all_labels.extend([0] * len(ethnic_files))  # Label 0 for others
                all_ethnic_groups.extend([ethnic_name] * len(ethnic_files))
                total_negatives += len(ethnic_files)
            else:
                print(f"Warning: {ethnic_name} directory not found: {ethnic_dir}")
        
        print(f"\nDataset composition:")
        print(f"  YangMi (positive): {len(yangmi_files)}")
        print(f"  Others (negative): {total_negatives}")
        print(f"  Total samples: {len(all_files)}")
        
        # Balance the dataset if needed
        if total_negatives > len(yangmi_files) * 2:
            print("Balancing dataset by reducing negative samples...")
            # Keep all YangMi images but limit negatives
            max_negatives = len(yangmi_files) * 2
            
            # Randomly sample negatives while maintaining ethnic diversity
            negative_indices = [i for i, label in enumerate(all_labels) if label == 0]
            selected_negatives = np.random.choice(negative_indices, min(max_negatives, len(negative_indices)), replace=False)
            
            # Keep all positives and selected negatives
            positive_indices = [i for i, label in enumerate(all_labels) if label == 1]
            keep_indices = list(positive_indices) + list(selected_negatives)
            
            all_files = [all_files[i] for i in keep_indices]
            all_labels = [all_labels[i] for i in keep_indices]
            all_ethnic_groups = [all_ethnic_groups[i] for i in keep_indices]
            
            print(f"Balanced dataset: {len(all_files)} total samples")
        
        # Split into train/validation with stratification
        train_files, val_files, train_labels, val_labels, train_ethnic, val_ethnic = train_test_split(
            all_files, all_labels, all_ethnic_groups,
            test_size=test_size, 
            random_state=42, 
            stratify=all_labels
        )
        
        print(f"\nTrain/Val split:")
        print(f"  Training samples: {len(train_files)} (YangMi: {sum(train_labels)}, Others: {len(train_labels) - sum(train_labels)})")
        print(f"  Validation samples: {len(val_files)} (YangMi: {sum(val_labels)}, Others: {len(val_labels) - sum(val_labels)})")
        
        # Create datasets
        train_dataset = AdaFaceMultiEthnicDataset(train_files, train_labels, train_ethnic, is_training=True)
        val_dataset = AdaFaceMultiEthnicDataset(val_files, val_labels, val_ethnic, is_training=False)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        return train_loader, val_loader
    
    def _get_image_files(self, directory):
        """Get all image files from directory"""
        if not os.path.exists(directory):
            return []
            
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob(os.path.join(directory, ext)))
            image_files.extend(glob(os.path.join(directory, ext.upper())))
        return image_files
    
    def freeze_backbone_layers(self, freeze_ratio=0.8):
        """Freeze early layers of the backbone for fine-tuning"""
        print(f"Freezing {freeze_ratio*100}% of backbone layers...")
        
        backbone_params = list(self.backbone.named_parameters())
        num_layers = len(backbone_params)
        freeze_until = int(num_layers * freeze_ratio)
        
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
    
    def train_model(self, train_loader, val_loader, num_epochs=20, learning_rate=0.0001):
        """Train AdaFace model with multi-ethnic data"""
        print("Starting AdaFace multi-ethnic training...")
        
        # Freeze backbone layers
        self.freeze_backbone_layers(freeze_ratio=0.8)
        
        # Setup optimizer with different learning rates
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        classifier_params = list(self.classifier.parameters())
        
        optimizer = optim.Adam([
            {'params': backbone_params, 'lr': learning_rate * 0.1},  # Lower LR for backbone
            {'params': classifier_params, 'lr': learning_rate}       # Higher LR for classifier
        ], weight_decay=1e-4)
        
        # Loss function with class weighting for imbalanced data
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
            ethnic_stats = {}
            
            for images, labels, ethnic_groups in tqdm(train_loader, desc="Training"):
                try:
                    images, labels = images.to(self.device), labels.to(self.device)
                    
                    if torch.any(torch.isnan(images)) or images.size(0) == 0:
                        continue
                    
                    # Track ethnic group statistics
                    for ethnic in ethnic_groups:
                        ethnic_stats[ethnic] = ethnic_stats.get(ethnic, 0) + 1
                    
                    optimizer.zero_grad()
                    
                    # Forward pass through AdaFace backbone
                    try:
                        features, norms = self.backbone(images)
                    except:
                        backbone_output = self.backbone(images)
                        if isinstance(backbone_output, tuple):
                            features = backbone_output[0]
                        else:
                            features = backbone_output
                    
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
            val_predictions = {'yangmi_correct': 0, 'others_correct': 0, 'yangmi_total': 0, 'others_total': 0}
            
            with torch.no_grad():
                for images, labels, ethnic_groups in tqdm(val_loader, desc="Validation"):
                    try:
                        images, labels = images.to(self.device), labels.to(self.device)
                        
                        if torch.any(torch.isnan(images)) or images.size(0) == 0:
                            continue
                        
                        # Forward pass
                        try:
                            features, norms = self.backbone(images)
                        except:
                            backbone_output = self.backbone(images)
                            if isinstance(backbone_output, tuple):
                                features = backbone_output[0]
                            else:
                                features = backbone_output
                        
                        outputs = self.classifier(features)
                        loss = criterion(outputs, labels)
                        
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()
                        
                        # Detailed accuracy by class
                        for i in range(len(labels)):
                            if labels[i] == 1:  # YangMi
                                val_predictions['yangmi_total'] += 1
                                if predicted[i] == 1:
                                    val_predictions['yangmi_correct'] += 1
                            else:  # Others
                                val_predictions['others_total'] += 1
                                if predicted[i] == 0:
                                    val_predictions['others_correct'] += 1
                        
                    except Exception as e:
                        print(f"Validation error: {e}")
                        continue
            
            # Calculate metrics
            train_acc = 100 * train_correct / train_total if train_total > 0 else 0
            val_acc = 100 * val_correct / val_total if val_total > 0 else 0
            
            # Detailed validation accuracy
            yangmi_acc = 100 * val_predictions['yangmi_correct'] / max(val_predictions['yangmi_total'], 1)
            others_acc = 100 * val_predictions['others_correct'] / max(val_predictions['others_total'], 1)
            
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
            print(f"  YangMi Acc: {yangmi_acc:.2f}% ({val_predictions['yangmi_correct']}/{val_predictions['yangmi_total']})")
            print(f"  Others Acc: {others_acc:.2f}% ({val_predictions['others_correct']}/{val_predictions['others_total']})")
            print(f"Training ethnic distribution: {ethnic_stats}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model('best_adaface_yangmi_multiethnic.pth')
                print(f"✓ New best model saved! Val Acc: {val_acc:.2f}%")
            
            scheduler.step()
        
        print(f"\nMulti-ethnic training completed! Best validation accuracy: {best_val_acc:.2f}%")
        return history
    
    def evaluate_on_test(self, test_dir, threshold=0.5):
        """Evaluate trained model on test images"""
        print(f"Evaluating on test images from {test_dir}")
        
        test_files = self._get_image_files(test_dir)
        if len(test_files) == 0:
            print(f"No test images found in {test_dir}")
            return []
        
        print(f"Found {len(test_files)} test images")
        
        results = []
        self.backbone.eval()
        self.classifier.eval()
        
        with torch.no_grad():
            for img_path in tqdm(test_files, desc="Processing test images"):
                try:
                    # Load and process image
                    temp_dataset = AdaFaceMultiEthnicDataset([img_path], [0], ['unknown'], is_training=False)
                    adaface_input, _, _ = temp_dataset[0]
                    
                    if torch.sum(adaface_input) == 0:
                        continue
                    
                    adaface_input = adaface_input.unsqueeze(0).to(self.device)
                    
                    # Get prediction
                    try:
                        features, norms = self.backbone(adaface_input)
                        feature_norm = norms[0].cpu().item()
                    except:
                        backbone_output = self.backbone(adaface_input)
                        if isinstance(backbone_output, tuple):
                            features = backbone_output[0]
                        else:
                            features = backbone_output
                        feature_norm = torch.norm(features).cpu().item()
                    
                    outputs = self.classifier(features)
                    probabilities = torch.softmax(outputs, dim=1)
                    _, predicted = torch.max(outputs, 1)
                    
                    # Store result
                    yangmi_prob = probabilities[0][1].cpu().item()
                    is_yangmi = predicted.cpu().item() == 1
                    confidence = max(probabilities[0]).cpu().item()
                    
                    results.append({
                        'test_image': os.path.basename(img_path),
                        'yangmi_probability': yangmi_prob,
                        'is_yangmi': is_yangmi,
                        'confidence': confidence,
                        'feature_norm': feature_norm,
                        'similarity': yangmi_prob,
                        'is_match': yangmi_prob > threshold,
                        'test_path': img_path
                    })
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    continue
        
        return results
    
    def save_model(self, filepath):
        """Save trained AdaFace model"""
        try:
            torch.save({
                'backbone_state_dict': self.backbone.state_dict(),
                'classifier_state_dict': self.classifier.state_dict(),
                'architecture': self.architecture
            }, filepath)
            print(f"Model saved to {filepath}")
        except Exception as e:
            print(f"Error saving model: {e}")

def main():
    """Main function for AdaFace multi-ethnic training"""
    print("=== AdaFace Multi-Ethnic Training for YangMi Recognition ===")
    
    # Define paths based on your EXACT directory structure
    yangmi_dir = 'elifiles/images/celeb-dataset/chinese/yangmi'
    
    # All the negative sample directories from your structure
    ethnic_dirs = {
        'caucasian_henrygolding': 'elifiles/images/celeb-dataset/caucasian/henrygolding',
        'caucasian_urassayasperbund': 'elifiles/images/celeb-dataset/caucasian/urassayasperbund',
        'chinese_gongli': 'elifiles/images/celeb-dataset/chinese/gongli',
        'chinese_ronnychieng': 'elifiles/images/celeb-dataset/chinese/ronnychieng',
        'indian_irrfankhan': 'elifiles/images/celeb-dataset/indian/irrfankhan',
        'indian_priyankachopra': 'elifiles/images/celeb-dataset/indian/priyankachopra',
        'malay_aaronaziz': 'elifiles/images/celeb-dataset/malay/aaronaziz',
        'malay_sitinurhaliza': 'elifiles/images/celeb-dataset/malay/sitinurhaliza'
    }
    
    # Test directories (if they exist)
    test_dirs = [
        'elifiles/images/celeb-dataset/chinese/yangmi_test',
        'elifiles/images/celeb-dataset/caucasian/henrygolding_test',
        'elifiles/images/celeb-dataset/caucasian/urassayasperbund_test',
        'elifiles/images/celeb-dataset/chinese/gongli_test',
        'elifiles/images/celeb-dataset/chinese/ronnychieng_test',
        'elifiles/images/celeb-dataset/indian/irrfankhan_test', 
        'elifiles/images/celeb-dataset/indian/priyankachopra_test',
        'elifiles/images/celeb-dataset/malay/aaronaziz_test',
        'elifiles/images/celeb-dataset/malay/sitinurhaliza_test'
    ]
    
    # Check if YangMi directory exists
    if not os.path.exists(yangmi_dir):
        print(f"YangMi directory not found: {yangmi_dir}")
        print("Please check your directory structure")
        return
    
    # Show available directories
    print("Dataset structure detected:")
    print(f"✓ YangMi (target): {yangmi_dir}")
    
    available_negatives = []
    for name, path in ethnic_dirs.items():
        if os.path.exists(path):
            count = len(trainer._get_image_files(path)) if 'trainer' in locals() else 'N/A'
            print(f"✓ {name}: {path}")
            available_negatives.append(name)
        else:
            print(f"✗ {name}: {path} (not found)")
    
    if len(available_negatives) == 0:
        print("ERROR: No negative sample directories found!")
        return
    
    # Initialize trainer
    try:
        trainer = AdaFaceYangMiMultiEthnicTrainer(
            architecture='ir_50',
            pretrained_path='pretrained/adaface_ir50_ms1mv2.ckpt'
        )
    except Exception as e:
        print(f"Error initializing trainer: {e}")
        return
    
    # Update available negatives count
    print(f"\nAvailable negative sample groups: {len(available_negatives)}")
    for name, path in ethnic_dirs.items():
        if os.path.exists(path):
            count = len(trainer._get_image_files(path))
            print(f"  {name}: {count} images")
    
    try:
        print("\n=== Preparing Multi-Celebrity Training Data ===")
        train_loader, val_loader = trainer.prepare_multiethnic_data(
            yangmi_dir, ethnic_dirs, batch_size=8
        )
        
        if len(train_loader) == 0:
            print("ERROR: No training data found!")
            return
        
        print("\n=== Training AdaFace Model with Multi-Celebrity Data ===")
        history = trainer.train_model(train_loader, val_loader, num_epochs=15)
        
        # Test the trained model on all available test directories
        print("\n=== Evaluating Trained Model ===")
        all_results = []
        
        for test_dir in test_dirs:
            if os.path.exists(test_dir):
                print(f"\nTesting on: {test_dir}")
                results = trainer.evaluate_on_test(test_dir)
                
                if results:
                    # Add source info to results
                    for result in results:
                        result['test_source'] = os.path.basename(test_dir)
                        result['expected_yangmi'] = 'yangmi' in test_dir.lower()
                    
                    all_results.extend(results)
                    
                    # Quick summary for this test set
                    yangmi_detected = sum(1 for r in results if r['is_yangmi'])
                    expected_yangmi = 'yangmi' in test_dir.lower()
                    print(f"  {test_dir}: {yangmi_detected}/{len(results)} classified as YangMi (Expected YangMi: {expected_yangmi})")
        
        if all_results:
            print(f"\n=== COMPREHENSIVE TEST RESULTS ===")
            
            # Create detailed results
            df = pd.DataFrame(all_results)
            df.to_csv('adaface_comprehensive_results.csv', index=False)
            
            # Overall statistics
            total_tested = len(all_results)
            total_yangmi_detected = sum(1 for r in all_results if r['is_yangmi'])
            
            # YangMi test performance (true positives)
            yangmi_tests = [r for r in all_results if r['expected_yangmi']]
            yangmi_detected_correctly = sum(1 for r in yangmi_tests if r['is_yangmi'])
            yangmi_recall = yangmi_detected_correctly / len(yangmi_tests) if yangmi_tests else 0
            
            # Non-YangMi test performance (true negatives) 
            non_yangmi_tests = [r for r in all_results if not r['expected_yangmi']]
            non_yangmi_rejected_correctly = sum(1 for r in non_yangmi_tests if not r['is_yangmi'])
            non_yangmi_precision = non_yangmi_rejected_correctly / len(non_yangmi_tests) if non_yangmi_tests else 0
            
            print(f"Total images tested: {total_tested}")
            print(f"Total classified as YangMi: {total_yangmi_detected}")
            print(f"YangMi Recall (True Positives): {yangmi_recall:.2%} ({yangmi_detected_correctly}/{len(yangmi_tests)})")
            print(f"Non-YangMi Precision (True Negatives): {non_yangmi_precision:.2%} ({non_yangmi_rejected_correctly}/{len(non_yangmi_tests)})")
            
            # Per-celebrity breakdown
            print(f"\nPer-celebrity performance:")
            for test_dir in test_dirs:
                if os.path.exists(test_dir):
                    celebrity_results = [r for r in all_results if r['test_source'] == os.path.basename(test_dir)]
                    if celebrity_results:
                        detected = sum(1 for r in celebrity_results if r['is_yangmi'])
                        avg_prob = np.mean([r['yangmi_probability'] for r in celebrity_results])
                        expected = celebrity_results[0]['expected_yangmi']
                        status = "✓" if (detected > len(celebrity_results)/2) == expected else "✗"
                        print(f"  {status} {os.path.basename(test_dir)}: {detected}/{len(celebrity_results)} as YangMi (avg prob: {avg_prob:.3f})")
            
            print(f"\nDetailed results saved to 'adaface_comprehensive_results.csv'")
        else:
            print("No test directories found or no valid test results")
        
        print("\n=== Multi-Celebrity Training Complete ===")
        print("Check 'best_adaface_yangmi_multiethnic.pth' for the trained model")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()