"""
Fruit Quality Assessment System - Main Pipeline
Novel approach combining texture, shape, and deep learning features
"""

import numpy as np
import cv2
from pathlib import Path
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier
import warnings
warnings.filterwarnings('ignore')

from feature_extraction import FeatureExtractor
from deep_learning_model import DeepLearningFeatures
from ensemble_classifier import EnsembleQualityClassifier
from data_preprocessing import ImagePreprocessor


class FruitQualityAssessmentSystem:
    """
    Complete fruit quality assessment pipeline
    """
    
    QUALITY_GRADES = {
        'A': {'range': (0.90, 1.0), 'description': 'Perfect - No defects'},
        'B': {'range': (0.75, 0.89), 'description': 'Good - Minor defects'},
        'C': {'range': (0.60, 0.74), 'description': 'Fair - Moderate defects'},
        'D': {'range': (0.0, 0.59), 'description': 'Poor - Rejected'}
    }
    
    def __init__(self, model_path=None, gpu_enabled=True):
        """Initialize the assessment system"""
        self.preprocessor = ImagePreprocessor()
        self.feature_extractor = FeatureExtractor()
        self.dl_features = DeepLearningFeatures(gpu_enabled=gpu_enabled)
        self.classifier = None
        self.scaler = StandardScaler()
        self.model_path = model_path
        
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
    
    def extract_combined_features(self, image_path):
        """
        Extract combined features: texture, shape, and deep learning
        
        Returns:
            dict: Combined feature vector with descriptions
        """
        # Load and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        processed_image, mask = self.preprocessor.preprocess(image)
        
        # Extract traditional features
        texture_features = self.feature_extractor.extract_texture_features(processed_image)
        shape_features = self.feature_extractor.extract_shape_features(processed_image, mask)
        color_features = self.feature_extractor.extract_color_features(processed_image)
        
        # Extract deep learning features
        dl_features = self.dl_features.extract_features(processed_image)
        
        # Combine all features
        combined_features = {
            'texture': texture_features,
            'shape': shape_features,
            'color': color_features,
            'deep_learning': dl_features,
            'image': processed_image,
            'mask': mask
        }
        
        return combined_features
    
    def train(self, image_dir, labels_dict, epochs=10, val_split=0.2):
        """
        Train the ensemble classifier
        
        Args:
            image_dir: Directory containing training images
            labels_dict: Dict mapping image names to quality grades
            epochs: Number of training epochs
            val_split: Validation split ratio
        """
        print("=" * 60)
        print("FRUIT QUALITY ASSESSMENT - TRAINING PHASE")
        print("=" * 60)
        
        image_paths = list(Path(image_dir).glob('*.jpg')) + \
                     list(Path(image_dir).glob('*.png'))
        
        X_train = []
        y_train = []
        
        print(f"\nExtracting features from {len(image_paths)} images...")
        
        for idx, img_path in enumerate(image_paths, 1):
            try:
                img_name = img_path.name
                if img_name not in labels_dict:
                    continue
                
                features = self.extract_combined_features(str(img_path))
                
                # Combine all feature vectors
                feature_vector = np.concatenate([
                    features['texture'],
                    features['shape'],
                    features['color'],
                    features['deep_learning']
                ])
                
                X_train.append(feature_vector)
                y_train.append(labels_dict[img_name])
                
                if idx % 50 == 0:
                    print(f"  Processed {idx} images...")
            
            except Exception as e:
                print(f"  Warning: Could not process {img_path}: {e}")
                continue
        
        if len(X_train) == 0:
            raise ValueError("No valid training data found")
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        print(f"\nFeature matrix shape: {X_train.shape}")
        print(f"Class distribution: {np.unique(y_train, return_counts=True)}")
        
        # Normalize features
        X_train = self.scaler.fit_transform(X_train)
        
        # Train ensemble classifier
        print("\nTraining ensemble classifier...")
        self.classifier = EnsembleQualityClassifier()
        self.classifier.train(X_train, y_train)
        
        print("✓ Training completed successfully!")
        return self.classifier
    
    def predict(self, image_path):
        """
        Predict quality grade for a single image
        
        Returns:
            dict: Quality grade, confidence, and detailed metrics
        """
        if self.classifier is None:
            raise ValueError("Model not trained. Call train() first or load a saved model.")
        
        # Extract features
        features = self.extract_combined_features(image_path)
        
        # Combine features
        feature_vector = np.concatenate([
            features['texture'],
            features['shape'],
            features['color'],
            features['deep_learning']
        ])
        
        # Normalize
        feature_vector = self.scaler.transform([feature_vector])[0]
        
        # Predict
        quality_score, confidence, predictions = self.classifier.predict(
            [feature_vector]
        )
        
        quality_score = quality_score[0]
        confidence = confidence[0]
        
        # Determine grade
        grade = self._get_grade_from_score(quality_score)
        
        result = {
            'image_path': image_path,
            'quality_score': float(quality_score),
            'confidence': float(confidence),
            'grade': grade,
            'grade_description': self.QUALITY_GRADES[grade]['description'],
            'predictions': predictions,
            'features_analysis': {
                'texture_score': float(np.mean(features['texture'])),
                'shape_score': float(np.mean(features['shape'])),
                'color_score': float(np.mean(features['color']))
            }
        }
        
        return result
    
    def batch_predict(self, image_dir, output_csv=None):
        """
        Predict quality for multiple images
        
        Returns:
            list: List of prediction results
        """
        image_paths = list(Path(image_dir).glob('*.jpg')) + \
                     list(Path(image_dir).glob('*.png'))
        
        results = []
        
        print(f"\nProcessing {len(image_paths)} images...")
        
        for idx, img_path in enumerate(image_paths, 1):
            try:
                result = self.predict(str(img_path))
                results.append(result)
                
                if idx % 10 == 0:
                    print(f"  Completed {idx}/{len(image_paths)}")
            
            except Exception as e:
                print(f"  Error processing {img_path}: {e}")
                continue
        
        # Save to CSV if requested
        if output_csv:
            self._save_results_to_csv(results, output_csv)
        
        return results
    
    def _get_grade_from_score(self, score):
        """Determine quality grade from score"""
        for grade, info in self.QUALITY_GRADES.items():
            if info['range'][0] <= score <= info['range'][1]:
                return grade
        return 'D'
    
    def save_model(self, filepath):
        """Save trained model"""
        if self.classifier is None:
            raise ValueError("No model to save. Train a model first.")
        
        model_data = {
            'classifier': self.classifier,
            'scaler': self.scaler,
            'quality_grades': self.QUALITY_GRADES
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✓ Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.classifier = model_data['classifier']
        self.scaler = model_data['scaler']
        
        print(f"✓ Model loaded from {filepath}")
    
    def _save_results_to_csv(self, results, filepath):
        """Save predictions to CSV"""
        import pandas as pd
        
        data = []
        for result in results:
            data.append({
                'image': Path(result['image_path']).name,
                'quality_score': result['quality_score'],
                'confidence': result['confidence'],
                'grade': result['grade'],
                'description': result['grade_description'],
                'texture_score': result['features_analysis']['texture_score'],
                'shape_score': result['features_analysis']['shape_score'],
                'color_score': result['features_analysis']['color_score']
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        print(f"✓ Results saved to {filepath}")
    
    def get_system_info(self):
        """Get system information"""
        return {
            'model_type': 'Ensemble (SVM + RF + XGBoost + MLP)',
            'feature_dimensions': {
                'texture': 52,
                'shape': 21,
                'color': 256,
                'deep_learning': 1280,
                'total': 1609
            },
            'quality_grades': self.QUALITY_GRADES,
            'gpu_enabled': self.dl_features.gpu_enabled
        }


if __name__ == "__main__":
    # Example usage
    system = FruitQualityAssessmentSystem(gpu_enabled=True)
    
    print("\nSystem Information:")
    info = system.get_system_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # To use:
    # 1. Train: system.train('path/to/training/images', labels_dict)
    # 2. Predict: result = system.predict('path/to/image.jpg')
    # 3. Batch: results = system.batch_predict('path/to/images/')
    # 4. Save: system.save_model('model.pkl')
