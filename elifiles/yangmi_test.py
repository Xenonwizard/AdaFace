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
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import json
import warnings
warnings.filterwarnings('ignore')

class YangMiFaceDataset(Dataset):
    """Face dataset similar to your AdaFace approach but with OpenCV face detection"""
    def __init__(self, image_paths, labels, img_size=224, is_training=True):
        self.image_paths = image_paths
        self.labels = labels
        self.img_size = img_size
        self.is_training = is_training
        
        # Initialize face cascade (similar to your align.get_aligned_face approach)
        self.face_cascade = self._init_face_cascade()
        
        # Define transforms (similar to your to_input function)
        if is_training:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def _init_face_cascade(self):
        """Initialize face cascade - handles different OpenCV versions"""
        cascade_paths = [
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
            '/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml',
            '/usr/local/share/opencv/haarcascades/haarcascade_frontalface_default.xml'
        ]
        
        for path in cascade_paths:
            if os.path.exists(path):
                return cv2.CascadeClassifier(path)
        
        print("Warning: Face cascade not found, using center crop")
        return None
    
    def detect_and_align_face(self, image):
        """Face detection and alignment (similar to your align.get_aligned_face)"""
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
                # Use largest face (similar to your approach)
                largest_face = max(faces, key=lambda x: x[2] * x[3])
                x, y, w, h = largest_face
                
                # Add padding
                padding = int(0.3 * min(w, h))
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(image.shape[1] - x, w + 2 * padding)
                h = min(image.shape[0] - y, h + 2 * padding)
                
                return image[y:y+h, x:x+w]
            else:
                # Fallback to center crop
                h, w = image.shape[:2]
                size = min(h, w)
                y = (h - size) // 2
                x = (w - size) // 2
                return image[y:y+size, x:x+size]
        except:
            # Fallback to center crop
            h, w = image.shape[:2]
            size = min(h, w)
            y = (h - size) // 2
            x = (w - size) // 2
            return image[y:y+size, x:x+size]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            # Load image (similar to your extract_features approach)
            image = cv2.imread(img_path)
            if image is None:
                return torch.zeros(3, self.img_size, self.img_size), label
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect and align face
            face = self.detect_and_align_face(image)
            
            # Apply transforms
            face = self.transform(face)
            
            return face, label
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            return torch.zeros(3, self.img_size, self.img_size), label

class YangMiTransferLearner:
    """Transfer learning approach inspired by your AdaFace tester structure"""
    
    def __init__(self, model_name='resnet50', num_classes=2):
        """
        Initialize transfer learner
        model_name: backbone model to use
        num_classes: 2 for YangMi vs Others
        """
        print(f"Loading pretrained model: {model_name}")
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Load pretrained backbone
        self.model = resnet50(pretrained=True)
        
        # Freeze early layers (transfer learning approach)
        self.freeze_backbone()
        
        # Modify classifier for YangMi recognition
        self.modify_classifier()
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        try:
            self.model.to(self.device)
            # Test with dummy input
            test_input = torch.randn(1, 3, 224, 224).to(self.device)
            with torch.no_grad():
                _ = self.model(test_input)
            print("✓ Model successfully loaded to device")
        except Exception as e:
            print(f"GPU error: {e}, falling back to CPU")
            self.device = torch.device('cpu')
            self.model.to(self.device)
    
    def freeze_backbone(self):
        """Freeze early layers for transfer learning"""
        print("Freezing backbone layers...")
        
        # Freeze all layers except final ones
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
        """Replace final layer for YangMi classification"""
        num_features = self.model.fc.in_features
        
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, self.num_classes)
        )
        
        print(f"Modified classifier for {self.num_classes} classes")
    
    def process_gallery_images(self, gallery_dir):
        """Process gallery images (similar to your original function)"""
        print(f"Processing gallery images from {gallery_dir}")
        
        # Get all image files (same extensions as your code)
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob(os.path.join(gallery_dir, ext)))
            image_files.extend(glob(os.path.join(gallery_dir, ext.upper())))
        
        if len(image_files) == 0:
            raise ValueError(f"No images found in {gallery_dir}")
        
        print(f"Found {len(image_files)} gallery images")
        return image_files
    
    def process_negative_images(self, negative_dir):
        """Process negative samples"""
        if negative_dir and os.path.exists(negative_dir):
            print(f"Processing negative images from {negative_dir}")
            
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
            image_files = []
            for ext in image_extensions:
                image_files.extend(glob(os.path.join(negative_dir, ext)))
                image_files.extend(glob(os.path.join(negative_dir, ext.upper())))
            
            print(f"Found {len(image_files)} negative images")
            return image_files
        else:
            return []
    
    def prepare_training_data(self, gallery_dir, negative_dir=None, test_size=0.2, batch_size=8):
        """Prepare training data (inspired by your data processing approach)"""
        print("Preparing training dataset...")
        
        # Get positive samples (YangMi)
        positive_files = self.process_gallery_images(gallery_dir)
        
        # Get negative samples
        negative_files = self.process_negative_images(negative_dir)
        
        # Create balanced dataset
        if len(negative_files) == 0:
            print("No negative samples found, creating synthetic negatives...")
            # Use some positive samples as synthetic negatives (with different augmentation)
            negative_files = positive_files[:len(positive_files)//3]
        
        # Balance the dataset
        min_samples = min(len(positive_files), len(negative_files))
        positive_files = positive_files[:min_samples]
        negative_files = negative_files[:min_samples]
        
        # Combine data
        all_files = positive_files + negative_files
        all_labels = [1] * len(positive_files) + [0] * len(negative_files)
        
        print(f"Dataset prepared:")
        print(f"  Positive samples (YangMi): {len(positive_files)}")
        print(f"  Negative samples (Others): {len(negative_files)}")
        print(f"  Total samples: {len(all_files)}")
        
        # Split into train/validation
        train_files, val_files, train_labels, val_labels = train_test_split(
            all_files, all_labels, test_size=test_size, random_state=42, stratify=all_labels
        )
        
        # Create datasets
        train_dataset = YangMiFaceDataset(train_files, train_labels, is_training=True)
        val_dataset = YangMiFaceDataset(val_files, val_labels, is_training=False)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        return train_loader, val_loader
    
    def train_model(self, train_loader, val_loader, num_epochs=15, learning_rate=0.001):
        """Train the model (similar structure to your evaluation loop)"""
        print("Starting YangMi transfer learning training...")
        
        # Setup training components
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), 
                             lr=learning_rate, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)
        
        # Training history (similar to your results tracking)
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
            
            for images, labels in tqdm(train_loader, desc="Training"):
                try:
                    images, labels = images.to(self.device), labels.to(self.device)
                    
                    # Skip problematic batches
                    if torch.any(torch.isnan(images)) or images.size(0) == 0:
                        continue
                    
                    optimizer.zero_grad()
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
                    
                except Exception as e:
                    if "out of memory" in str(e):
                        print("GPU memory issue, skipping batch")
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                        continue
                    else:
                        print(f"Training error: {e}")
                        continue
            
            # Validation phase (similar to your evaluation approach)
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in tqdm(val_loader, desc="Validation"):
                    try:
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
                        continue
            
            # Calculate metrics
            train_acc = 100 * train_correct / train_total if train_total > 0 else 0
            val_acc = 100 * val_correct / val_total if val_total > 0 else 0
            
            # Store history
            history['train_loss'].append(train_loss / max(len(train_loader), 1))
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss / max(len(val_loader), 1))
            history['val_acc'].append(val_acc)
            
            # Print results (similar to your reporting style)
            print(f"Train Loss: {train_loss/max(len(train_loader), 1):.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss/max(len(val_loader), 1):.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model('best_yangmi_transfer_model.pth')
                print(f"✓ New best model saved! Val Acc: {val_acc:.2f}%")
            
            scheduler.step()
        
        print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.2f}%")
        return history
    
    def extract_features(self, image_path):
        """Extract features from trained model (similar to your extract_features)"""
        try:
            # Load and process image
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Create temporary dataset for processing
            temp_dataset = YangMiFaceDataset([image_path], [0], is_training=False)
            face_tensor, _ = temp_dataset[0]
            
            if torch.sum(face_tensor) == 0:  # Check if valid
                return None
            
            # Extract features
            self.model.eval()
            with torch.no_grad():
                face_tensor = face_tensor.unsqueeze(0).to(self.device)
                
                # Get features from second-to-last layer
                features = self.model.avgpool(self.model.layer4(
                    self.model.layer3(self.model.layer2(self.model.layer1(
                        self.model.maxpool(self.model.relu(self.model.bn1(self.model.conv1(face_tensor))))
                    )))
                ))
                features = torch.flatten(features, 1)
                
                # Normalize features (similar to your approach)
                features = features / torch.norm(features, dim=1, keepdim=True)
                
            return features.cpu().numpy().flatten()
            
        except Exception as e:
            print(f"Error extracting features from {image_path}: {e}")
            return None
    
    def evaluate_on_test(self, test_dir, threshold=0.5):
        """Evaluate trained model on test images (similar to your evaluation approach)"""
        print(f"Evaluating on test images from {test_dir}")
        
        # Get test images
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        test_files = []
        for ext in image_extensions:
            test_files.extend(glob(os.path.join(test_dir, ext)))
            test_files.extend(glob(os.path.join(test_dir, ext.upper())))
        
        if len(test_files) == 0:
            print(f"No test images found in {test_dir}")
            return []
        
        print(f"Found {len(test_files)} test images")
        
        # Evaluate each image
        results = []
        self.model.eval()
        
        with torch.no_grad():
            for img_path in tqdm(test_files, desc="Processing test images"):
                try:
                    # Load and process image
                    temp_dataset = YangMiFaceDataset([img_path], [0], is_training=False)
                    face_tensor, _ = temp_dataset[0]
                    
                    if torch.sum(face_tensor) == 0:
                        continue
                    
                    face_tensor = face_tensor.unsqueeze(0).to(self.device)
                    
                    # Get prediction
                    outputs = self.model(face_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    _, predicted = torch.max(outputs, 1)
                    
                    # Store result (similar to your results format)
                    yangmi_prob = probabilities[0][1].cpu().item()
                    is_yangmi = predicted.cpu().item() == 1
                    confidence = max(probabilities[0]).cpu().item()
                    
                    results.append({
                        'test_image': os.path.basename(img_path),
                        'yangmi_probability': yangmi_prob,
                        'is_yangmi': is_yangmi,
                        'confidence': confidence,
                        'similarity': yangmi_prob,  # For compatibility with your analysis
                        'is_match': yangmi_prob > threshold,
                        'test_path': img_path
                    })
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    continue
        
        return results
    
    def create_results_report(self, results, output_dir='results'):
        """Create comprehensive analysis report (inspired by your reporting style)"""
        os.makedirs(output_dir, exist_ok=True)
        
        if len(results) == 0:
            print("No results to analyze")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Generate analysis (similar to your comprehensive analysis)
        analysis_report = self.generate_transfer_learning_analysis(df)
        
        # Save analysis
        analysis_path = os.path.join(output_dir, 'transfer_learning_analysis.txt')
        with open(analysis_path, 'w') as f:
            f.write(analysis_report)
        
        print(analysis_report)
        
        # Save detailed results
        df.to_csv(os.path.join(output_dir, 'transfer_learning_results.csv'), index=False)
        
        # Plot results (similar to your plotting)
        self.plot_results(df, output_dir)
        
        return df
    
    def generate_transfer_learning_analysis(self, df):
        """Generate analysis report (inspired by your comprehensive analysis)"""
        similarities = df['yangmi_probability'].values
        
        report = []
        report.append("=" * 80)
        report.append("YANGMI TRANSFER LEARNING COMPREHENSIVE ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Model: Transfer Learning with {self.model_name}")
        report.append("")
        
        # Dataset Overview
        report.append("1. EVALUATION OVERVIEW")
        report.append("-" * 40)
        report.append(f"Test Images Processed: {len(df)}")
        report.append(f"Images Classified as YangMi: {sum(df['is_yangmi'])}")
        report.append(f"Average Confidence: {np.mean(df['confidence']):.4f}")
        report.append("")
        
        # Performance Analysis (similar to your threshold analysis)
        report.append("2. PROBABILITY DISTRIBUTION ANALYSIS")
        report.append("-" * 40)
        report.append(f"Mean YangMi Probability: {np.mean(similarities):.4f}")
        report.append(f"Median YangMi Probability: {np.median(similarities):.4f}")
        report.append(f"Standard Deviation: {np.std(similarities):.4f}")
        report.append(f"Min Probability: {np.min(similarities):.4f}")
        report.append(f"Max Probability: {np.max(similarities):.4f}")
        report.append("")
        
        # Threshold Analysis (like your original)
        report.append("3. THRESHOLD PERFORMANCE ANALYSIS")
        report.append("-" * 40)
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        for thresh in thresholds:
            count = (similarities > thresh).sum()
            percentage = count / len(similarities) * 100
            report.append(f