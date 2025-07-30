import torch
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

class AdaFaceYangMiTester:
    def __init__(self, model_name='ir_50'):
        """
        Initialize AdaFace tester
        model_name: 'ir_18', 'ir_50', 'ir_100' - choose best pretrained model
        """
        print(f"Loading AdaFace model: {model_name}")
        self.model = load_pretrained_model(model_name)
        self.model.eval()
        
    def extract_features(self, image_path):
        """Extract features from a single image"""
        try:
            # Align face using MTCNN
            aligned_rgb_img = align.get_aligned_face(image_path)
            if aligned_rgb_img is None:
                print(f"No face detected in {image_path}")
                return None
                
            # Convert to model input format (BGR, normalized)
            bgr_input = to_input(aligned_rgb_img)
            
            # Extract features
            with torch.no_grad():
                feature, _ = self.model(bgr_input)
                # Normalize features
                feature = feature / torch.norm(feature, dim=1, keepdim=True)
                
            return feature.cpu().numpy().flatten()
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return None
    
    def process_gallery_images(self, gallery_dir):
        """Process all images in gallery (training) folder"""
        print(f"Processing gallery images from {gallery_dir}")
        
        # Get all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob(os.path.join(gallery_dir, ext)))
            image_files.extend(glob(os.path.join(gallery_dir, ext.upper())))
        
        gallery_features = []
        gallery_paths = []
        
        for img_path in tqdm(image_files, desc="Processing gallery"):
            feature = self.extract_features(img_path)
            if feature is not None:
                gallery_features.append(feature)
                gallery_paths.append(img_path)
        
        if len(gallery_features) == 0:
            raise ValueError("No valid gallery images found!")
            
        return np.array(gallery_features), gallery_paths
    
    def process_test_images(self, test_dir):
        """Process all images in test folder"""
        print(f"Processing test images from {test_dir}")
        
        # Get all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob(os.path.join(test_dir, ext)))
            image_files.extend(glob(os.path.join(test_dir, ext.upper())))
        
        test_features = []
        test_paths = []
        
        for img_path in tqdm(image_files, desc="Processing test images"):
            feature = self.extract_features(img_path)
            if feature is not None:
                test_features.append(feature)
                test_paths.append(img_path)
        
        if len(test_features) == 0:
            raise ValueError("No valid test images found!")
            
        return np.array(test_features), test_paths
    
    def compute_similarities(self, gallery_features, test_features):
        """Compute cosine similarities between test and gallery images"""
        return cosine_similarity(test_features, gallery_features)
    
    def evaluate_performance(self, similarities, test_paths, gallery_paths, threshold=0.5):
        """Evaluate recognition performance"""
        results = []
        
        for i, test_path in enumerate(test_paths):
            # Find most similar gallery image
            max_sim_idx = np.argmax(similarities[i])
            max_similarity = similarities[i][max_sim_idx]
            best_match = gallery_paths[max_sim_idx]
            
            # Simple evaluation: if similarity > threshold, consider it a match
            is_match = max_similarity > threshold
            
            results.append({
                'test_image': os.path.basename(test_path),
                'best_match': os.path.basename(best_match),
                'similarity': max_similarity,
                'is_match': is_match,
                'test_path': test_path,
                'match_path': best_match
            })
        
        return results
    
    def create_results_report(self, results, gallery_features, test_features, output_dir='results'):
        """Create detailed results report and comprehensive analysis"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(results)
        
        # Generate comprehensive analysis
        analysis_report = self.generate_comprehensive_analysis(df, gallery_features, test_features)
        
        # Save analysis to file
        analysis_path = os.path.join(output_dir, 'comprehensive_analysis.txt')
        with open(analysis_path, 'w') as f:
            f.write(analysis_report)
        
        # Print analysis to console
        print(analysis_report)
        
        # Save detailed results
        df.to_csv(os.path.join(output_dir, 'detailed_results.csv'), index=False)
        
        # Save summary statistics as JSON
        import json
        summary_stats = self.calculate_summary_statistics(df, gallery_features, test_features)
        with open(os.path.join(output_dir, 'summary_statistics.json'), 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        # Plot similarity distribution
        plt.figure(figsize=(10, 6))
        plt.hist(df['similarity'], bins=30, alpha=0.7, edgecolor='black')
        plt.axvline(x=0.5, color='red', linestyle='--', label='Threshold (0.5)')
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Frequency')
        plt.title('Distribution of Similarities - YangMi AdaFace Test')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'similarity_distribution.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        return df
    
    def calculate_summary_statistics(self, df, gallery_features, test_features):
        """Calculate comprehensive summary statistics"""
        similarities = df['similarity'].values
        
        stats = {
            'dataset_info': {
                'total_gallery_images': len(gallery_features),
                'total_test_images': len(test_features),
                'successfully_processed_test_images': len(df)
            },
            'similarity_statistics': {
                'mean_similarity': float(np.mean(similarities)),
                'median_similarity': float(np.median(similarities)),
                'std_similarity': float(np.std(similarities)),
                'min_similarity': float(np.min(similarities)),
                'max_similarity': float(np.max(similarities)),
                'q25_similarity': float(np.percentile(similarities, 25)),
                'q75_similarity': float(np.percentile(similarities, 75))
            },
            'threshold_analysis': {
                'threshold_0.3': int((similarities > 0.3).sum()),
                'threshold_0.4': int((similarities > 0.4).sum()),
                'threshold_0.5': int((similarities > 0.5).sum()),
                'threshold_0.6': int((similarities > 0.6).sum()),
                'threshold_0.7': int((similarities > 0.7).sum()),
                'threshold_0.8': int((similarities > 0.8).sum()),
                'threshold_0.9': int((similarities > 0.9).sum())
            },
            'performance_metrics': {
                'accuracy_at_0.5': float((similarities > 0.5).sum() / len(similarities)),
                'accuracy_at_0.6': float((similarities > 0.6).sum() / len(similarities)),
                'accuracy_at_0.7': float((similarities > 0.7).sum() / len(similarities))
            }
        }
        
        return stats
    
    def generate_comprehensive_analysis(self, df, gallery_features, test_features):
        """Generate comprehensive analysis report"""
        similarities = df['similarity'].values
        
        report = []
        report.append("=" * 80)
        report.append("ADAFACE YANGMI DATASET COMPREHENSIVE ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Dataset Overview
        report.append("1. DATASET OVERVIEW")
        report.append("-" * 40)
        report.append(f"Gallery Images (Training): {len(gallery_features)}")
        report.append(f"Test Images: {len(test_features)}")
        report.append(f"Successfully Processed Test Images: {len(df)}")
        report.append(f"Processing Success Rate: {len(df)/len(test_features)*100:.1f}%")
        report.append("")
        
        # Similarity Statistics
        report.append("2. SIMILARITY DISTRIBUTION ANALYSIS")
        report.append("-" * 40)
        report.append(f"Mean Similarity: {np.mean(similarities):.4f}")
        report.append(f"Median Similarity: {np.median(similarities):.4f}")
        report.append(f"Standard Deviation: {np.std(similarities):.4f}")
        report.append(f"Minimum Similarity: {np.min(similarities):.4f}")
        report.append(f"Maximum Similarity: {np.max(similarities):.4f}")
        report.append(f"25th Percentile: {np.percentile(similarities, 25):.4f}")
        report.append(f"75th Percentile: {np.percentile(similarities, 75):.4f}")
        report.append("")
        
        # Threshold Analysis
        report.append("3. THRESHOLD PERFORMANCE ANALYSIS")
        report.append("-" * 40)
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        for thresh in thresholds:
            count = (similarities > thresh).sum()
            percentage = count / len(similarities) * 100
            report.append(f"Threshold {thresh}: {count}/{len(similarities)} ({percentage:.1f}%)")
        report.append("")
        
        # Performance Interpretation
        report.append("4. PERFORMANCE INTERPRETATION")
        report.append("-" * 40)
        avg_sim = np.mean(similarities)
        if avg_sim > 0.8:
            performance = "EXCELLENT"
            interpretation = "AdaFace shows exceptional performance on YangMi dataset"
        elif avg_sim > 0.7:
            performance = "VERY GOOD"
            interpretation = "AdaFace shows very good performance on YangMi dataset"
        elif avg_sim > 0.6:
            performance = "GOOD"
            interpretation = "AdaFace shows good performance on YangMi dataset"
        elif avg_sim > 0.5:
            performance = "MODERATE"
            interpretation = "AdaFace shows moderate performance on YangMi dataset"
        else:
            performance = "POOR"
            interpretation = "AdaFace shows poor performance on YangMi dataset"
        
        report.append(f"Overall Performance: {performance}")
        report.append(f"Interpretation: {interpretation}")
        report.append("")
        
        # Top and Bottom Performers
        report.append("5. TOP AND BOTTOM PERFORMING MATCHES")
        report.append("-" * 40)
        report.append("TOP 5 MATCHES:")
        top_5 = df.nlargest(5, 'similarity')[['test_image', 'best_match', 'similarity']]
        for idx, row in top_5.iterrows():
            report.append(f"  {row['test_image']} → {row['best_match']} (sim: {row['similarity']:.4f})")
        
        report.append("\nBOTTOM 5 MATCHES:")
        bottom_5 = df.nsmallest(5, 'similarity')[['test_image', 'best_match', 'similarity']]
        for idx, row in bottom_5.iterrows():
            report.append(f"  {row['test_image']} → {row['best_match']} (sim: {row['similarity']:.4f})")
        report.append("")
        
        # Quality Assessment
        report.append("6. QUALITY ASSESSMENT")
        report.append("-" * 40)
        high_quality = (similarities > 0.8).sum()
        medium_quality = ((similarities > 0.6) & (similarities <= 0.8)).sum()
        low_quality = (similarities <= 0.6).sum()
        
        report.append(f"High Quality Matches (>0.8): {high_quality} ({high_quality/len(similarities)*100:.1f}%)")
        report.append(f"Medium Quality Matches (0.6-0.8): {medium_quality} ({medium_quality/len(similarities)*100:.1f}%)")
        report.append(f"Low Quality Matches (≤0.6): {low_quality} ({low_quality/len(similarities)*100:.1f}%)")
        report.append("")
        
        # Recommendations
        report.append("7. RECOMMENDATIONS")
        report.append("-" * 40)
        if avg_sim > 0.7:
            report.append("✓ Model performs well on this dataset")
            report.append("✓ Consider using threshold 0.6-0.7 for practical applications")
        else:
            report.append("⚠ Consider collecting more diverse training images")
            report.append("⚠ Check image quality and lighting conditions")
            report.append("⚠ Consider fine-tuning the model on this specific dataset")
        
        if high_quality < len(similarities) * 0.5:
            report.append("⚠ Less than 50% high-quality matches - investigate data quality")
        
        report.append("")
        report.append("8. FILES GENERATED")
        report.append("-" * 40)
        report.append("- comprehensive_analysis.txt (this file)")
        report.append("- detailed_results.csv (all similarity scores)")
        report.append("- summary_statistics.json (machine-readable stats)")
        report.append("- similarity_distribution.png (histogram)")
        report.append("- top_matches_visualization.png (visual matches)")
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def visualize_matches(self, results, output_dir='results', top_n=5):
        """Visualize top matches"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Sort by similarity
        sorted_results = sorted(results, key=lambda x: x['similarity'], reverse=True)
        
        fig, axes = plt.subplots(top_n, 2, figsize=(12, 4*top_n))
        fig.suptitle('Top AdaFace Matches - YangMi Dataset', fontsize=16)
        
        for i in range(min(top_n, len(sorted_results))):
            result = sorted_results[i]
            
            # Load and display test image
            test_img = cv2.imread(result['test_path'])
            test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
            axes[i, 0].imshow(test_img)
            axes[i, 0].set_title(f"Test: {result['test_image']}")
            axes[i, 0].axis('off')
            
            # Load and display best match
            match_img = cv2.imread(result['match_path'])
            match_img = cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB)
            axes[i, 1].imshow(match_img)
            axes[i, 1].set_title(f"Match: {result['best_match']}\nSimilarity: {result['similarity']:.4f}")
            axes[i, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'top_matches_visualization.png'), dpi=300, bbox_inches='tight')
        plt.show()

def main():
    # Initialize tester with best pretrained model
    tester = AdaFaceYangMiTester(model_name='ir_100')  # Use the best model
    
    # Set paths
    gallery_dir = 'elifiles/images/yangmi'  # Training images
    test_dir = 'elifiles/images/yangmi_test'  # Test images
    
    print("Starting YangMi AdaFace evaluation...")
    
    try:
        # Process gallery (training) images
        gallery_features, gallery_paths = tester.process_gallery_images(gallery_dir)
        print(f"Gallery processed: {len(gallery_features)} valid images")
        
        # Process test images
        test_features, test_paths = tester.process_test_images(test_dir)
        print(f"Test set processed: {len(test_features)} valid images")
        
        # Compute similarities
        print("Computing similarities...")
        similarities = tester.compute_similarities(gallery_features, test_features)
        
        # Evaluate performance
        print("Evaluating performance...")
        results = tester.evaluate_performance(similarities, test_paths, gallery_paths, threshold=0.5)
        
        # Create results report
        df = tester.create_results_report(results, gallery_features, test_features)
        
        # Visualize top matches
        tester.visualize_matches(results, top_n=5)
        
        print(f"\nResults saved to 'results/' directory")
        print("Check 'detailed_results.csv' for full analysis")
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        print("Make sure you have:")
        print("1. Downloaded a pretrained AdaFace model")
        print("2. Images in the specified directories")
        print("3. All dependencies installed")

if __name__ == "__main__":
    main()