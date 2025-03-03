# core/model_trainer.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Class for training and evaluating defect prediction models."""
    
    def __init__(self, csv_path):
        """
        Initialize the model trainer with a CSV file.
        
        Args:
            csv_path (str): Path to the CSV file with metrics and defect labels
        """
        self.csv_path = csv_path
        self.data = None
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def load_data(self):
        """
        Load and preprocess the data from the CSV file.
        
        Returns:
            pandas.DataFrame: The loaded dataframe
        """
        try:
            self.data = pd.read_csv(self.csv_path)
            self.logger.info(f"Loaded data from {self.csv_path} with {len(self.data)} records")
            
            # Check if required columns exist
            if 'defective' not in self.data.columns:
                self.logger.error("CSV file does not contain 'defective' column")
                raise ValueError("CSV file must contain 'defective' column")
            
            # Set feature columns (all except file_path and defective)
            self.feature_columns = [col for col in self.data.columns 
                                  if col not in ['file_path', 'defective']]
            
            # Check if there are enough features
            if len(self.feature_columns) == 0:
                self.logger.error("No feature columns found in the CSV file")
                raise ValueError("CSV file must contain feature columns")
            
            # Replace NaN values with 0
            self.data = self.data.fillna(0)
            
            return self.data
        
        except Exception as e:
            self.logger.error(f"Error loading CSV data: {str(e)}")
            raise
    
    def split_data(self, test_size=0.2, random_state=42):
        """
        Split the data into training and testing sets.
        
        Args:
            test_size (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test) splits
        """
        if self.data is None:
            self.load_data()
        
        X = self.data[self.feature_columns]
        y = self.data['defective']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale the features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.logger.info(f"Data split into {len(X_train)} training samples and {len(X_test)} test samples")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_model(self, n_estimators=100, max_depth=None):
        """
        Train a RandomForest model on the data.
        
        Args:
            n_estimators (int): Number of trees in the forest
            max_depth (int): Maximum depth of the trees
            
        Returns:
            sklearn.ensemble.RandomForestClassifier: The trained model
        """
        X_train, X_test, y_train, y_test = self.split_data()
        
        self.logger.info(f"Training Random Forest model with {n_estimators} estimators")
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1  # Use all available cores
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        self.logger.info(f"Model trained successfully. Metrics: {metrics}")
        
        # Get feature importances
        feature_importance = dict(zip(self.feature_columns, self.model.feature_importances_))
        sorted_importance = {k: v for k, v in sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)}
        
        self.logger.info(f"Top 5 most important features: {list(sorted_importance.items())[:5]}")
        
        return self.model, metrics
    
    def cross_validate(self, cv=5):
        """
        Perform cross-validation on the model.
        
        Args:
            cv (int): Number of cross-validation folds
            
        Returns:
            dict: Cross-validation metrics
        """
        if self.data is None:
            self.load_data()
        
        X = self.data[self.feature_columns]
        y = self.data['defective']
        
        # Scale the features
        if self.scaler is None:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        
        self.logger.info(f"Performing {cv}-fold cross-validation")
        
        cv_accuracy = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        cv_precision = cross_val_score(model, X, y, cv=cv, scoring='precision')
        cv_recall = cross_val_score(model, X, y, cv=cv, scoring='recall')
        cv_f1 = cross_val_score(model, X, y, cv=cv, scoring='f1')
        
        cv_metrics = {
            'accuracy': {
                'mean': cv_accuracy.mean(),
                'std': cv_accuracy.std()
            },
            'precision': {
                'mean': cv_precision.mean(),
                'std': cv_precision.std()
            },
            'recall': {
                'mean': cv_recall.mean(),
                'std': cv_recall.std()
            },
            'f1_score': {
                'mean': cv_f1.mean(),
                'std': cv_f1.std()
            }
        }
        
        self.logger.info(f"Cross-validation metrics: {cv_metrics}")
        
        return cv_metrics
    
    def save_model(self, model_dir):
        """
        Save the trained model and scaler to disk.
        
        Args:
            model_dir (str): Directory to save the model
            
        Returns:
            tuple: Paths to the saved model and scaler
        """
        if self.model is None:
            self.logger.error("No trained model to save")
            raise ValueError("You must train a model before saving")
        
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, 'model.joblib')
        scaler_path = os.path.join(model_dir, 'scaler.joblib')
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        self.logger.info(f"Model saved to {model_path}")
        self.logger.info(f"Scaler saved to {scaler_path}")
        
        # Save feature columns list
        features_path = os.path.join(model_dir, 'features.joblib')
        joblib.dump(self.feature_columns, features_path)
        
        return model_path, scaler_path, features_path