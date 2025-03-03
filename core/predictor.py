# core/predictor.py

import pandas as pd
import joblib
import os
import logging
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Predictor:
    """Class for making defect predictions using a trained model."""
    
    def __init__(self, model_dir):
        """
        Initialize the predictor with a trained model.
        
        Args:
            model_dir (str): Directory containing the model, scaler, and features
        """
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.load_model()
    
    def load_model(self):
        """
        Load the trained model, scaler, and feature columns from disk.
        
        Returns:
            bool: True if loading was successful
        """
        try:
            model_path = os.path.join(self.model_dir, 'model.joblib')
            scaler_path = os.path.join(self.model_dir, 'scaler.joblib')
            features_path = os.path.join(self.model_dir, 'features.joblib')
            
            if not all(os.path.exists(path) for path in [model_path, scaler_path, features_path]):
                self.logger.error(f"Missing model files in {self.model_dir}")
                raise ValueError(f"Missing model files in {self.model_dir}")
            
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.feature_columns = joblib.load(features_path)
            
            self.logger.info(f"Model loaded successfully from {self.model_dir}")
            self.logger.info(f"Model has {len(self.feature_columns)} features")
            
            return True
        
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
    
    def predict(self, csv_path):
        """
        Make predictions for files in the given CSV.
        
        Args:
            csv_path (str): Path to CSV file with metrics
            
        Returns:
            pandas.DataFrame: The input data with added prediction columns
        """
        try:
            # Load the CSV data
            data = pd.read_csv(csv_path)
            self.logger.info(f"Loaded data from {csv_path} with {len(data)} records")
            
            # Check if file_path column exists
            if 'file_path' not in data.columns:
                self.logger.warning("CSV does not contain file_path column")
            
            # Ensure all required feature columns are present
            missing_features = [col for col in self.feature_columns if col not in data.columns]
            if missing_features:
                self.logger.error(f"Missing features in CSV: {missing_features}")
                raise ValueError(f"CSV is missing required features: {missing_features}")
            
            # Replace NaN values with 0
            data = data.fillna(0)
            
            # Extract features
            X = data[self.feature_columns]
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Make predictions
            predictions = self.model.predict(X_scaled)
            probabilities = self.model.predict_proba(X_scaled)[:, 1]  # Probability of defective class
            
            # Add predictions to the dataframe
            data['predicted_defective'] = predictions
            data['defect_probability'] = probabilities
            
            self.logger.info(f"Predicted {sum(predictions)} defective files out of {len(predictions)}")
            
            return data
        
        except Exception as e:
            self.logger.error(f"Error making predictions: {str(e)}")
            raise
    
    def compare_results(self, true_csv, pred_csv):
        """
        Compare prediction results with ground truth.
        
        Args:
            true_csv (str): Path to CSV with actual defects
            pred_csv (str): Path to CSV with predicted defects
            
        Returns:
            dict: Comparison metrics
        """
        try:
            # Load the CSVs
            true_data = pd.read_csv(true_csv)
            pred_data = pd.read_csv(pred_csv)
            
            # Ensure both have file_path and defective columns
            required_cols = ['file_path', 'defective']
            for df, path in [(true_data, true_csv), (pred_data, pred_csv)]:
                missing = [col for col in required_cols if col not in df.columns]
                if missing:
                    self.logger.error(f"Missing columns in {path}: {missing}")
                    raise ValueError(f"CSV is missing required columns: {missing}")
            
            # Merge on file_path
            merged = pd.merge(true_data, pred_data, on='file_path', suffixes=('_true', '_pred'))
            
            # Calculate metrics
            true_pos = sum((merged['defective_true'] == 1) & (merged['predicted_defective'] == 1))
            false_pos = sum((merged['defective_true'] == 0) & (merged['predicted_defective'] == 1))
            true_neg = sum((merged['defective_true'] == 0) & (merged['predicted_defective'] == 0))
            false_neg = sum((merged['defective_true'] == 1) & (merged['predicted_defective'] == 0))
            
            total = true_pos + false_pos + true_neg + false_neg
            accuracy = (true_pos + true_neg) / total if total > 0 else 0
            precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
            recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'true_positives': true_pos,
                'false_positives': false_pos,
                'true_negatives': true_neg,
                'false_negatives': false_neg
            }
            
            self.logger.info(f"Comparison metrics: {metrics}")
            
            return metrics
        
        except Exception as e:
            self.logger.error(f"Error comparing results: {str(e)}")
            raise