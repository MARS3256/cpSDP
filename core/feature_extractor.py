# core/feature_extractor.py

import os
import csv
import pandas as pd
from abc import ABC, abstractmethod
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureExtractor(ABC):
    """Base class for all language-specific feature extractors."""
    
    def __init__(self, project_path):
        """
        Initialize the feature extractor with the project path.
        
        Args:
            project_path (str): Path to the project root directory
        """
        self.project_path = project_path
        self.metrics = []
        self.file_extensions = []
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def extract_metrics_from_file(self, file_path):
        """
        Extract metrics from a single file.
        
        Args:
            file_path (str): Path to the file
            
        Returns:
            dict: Dictionary containing extracted metrics
        """
        pass
    
    def get_project_files(self):
        """
        Get all files in the project with relevant extensions.
        
        Returns:
            list: List of file paths
        """
        all_files = []
        for root, _, files in os.walk(self.project_path):
            for file in files:
                if any(file.endswith(ext) for ext in self.file_extensions):
                    all_files.append(os.path.join(root, file))
        
        self.logger.info(f"Found {len(all_files)} files with extensions {', '.join(self.file_extensions)}")
        return all_files
    
    def extract_project_metrics(self):
        """
        Extract metrics from all files in the project.
        
        Returns:
            list: List of dictionaries containing metrics for each file
        """
        project_files = self.get_project_files()
        metrics_data = []
        
        for file_path in project_files:
            try:
                self.logger.info(f"Processing file: {file_path}")
                metrics = self.extract_metrics_from_file(file_path)
                metrics['file_path'] = os.path.relpath(file_path, self.project_path)
                # Add default value of 0 for defective (to be labeled manually or predicted)
                metrics['defective'] = 0
                metrics_data.append(metrics)
            except Exception as e:
                self.logger.error(f"Error processing file {file_path}: {str(e)}")
        
        return metrics_data
    
    def save_metrics_to_csv(self, output_path):
        """
        Extract metrics from the project and save to CSV.
        
        Args:
            output_path (str): Path to save the CSV file
            
        Returns:
            str: Path to the saved CSV file
        """
        metrics_data = self.extract_project_metrics()
        
        if not metrics_data:
            self.logger.warning("No metrics data was extracted.")
            return None
        
        df = pd.DataFrame(metrics_data)
        df.to_csv(output_path, index=False)
        self.logger.info(f"Metrics saved to {output_path}")
        
        return output_path
    
    @staticmethod
    def get_all_metrics():
        """
        Get a list of all metrics being extracted.
        
        Returns:
            list: List of metric names
        """
        return [
            'CBO', 'DIT', 'rfc', 'totalMethods', 'totalFields', 'LOC',
            'returnQty', 'loopQty', 'comparisonsQty', 'tryCatchQty',
            'parenthesizedExpsQty', 'stringLiteralsQty', 'numbersQty',
            'assignmentsQty', 'mathOperationsQty', 'variablesQty',
            'maxNestedBlocks', 'uniqueWordsQty'
        ]
        
    def set_selected_directories(self, selected_dirs):
        """Set selected directories to scan."""
        self.selected_dirs = selected_dirs

    def get_project_files(self):
        """
        Get all files in the project with relevant extensions.
        
        Returns:
            list: List of file paths
        """
        all_files = []
        
        # Check if we have selected directories
        if hasattr(self, 'selected_dirs') and self.selected_dirs:
            # Only scan the selected directories
            for dir_path in self.selected_dirs:
                for root, _, files in os.walk(dir_path):
                    for file in files:
                        if any(file.endswith(ext) for ext in self.file_extensions):
                            all_files.append(os.path.join(root, file))
        else:
            # Default behavior - scan the whole project
            for root, _, files in os.walk(self.project_path):
                for file in files:
                    if any(file.endswith(ext) for ext in self.file_extensions):
                        all_files.append(os.path.join(root, file))
        
        self.logger.info(f"Found {len(all_files)} files with extensions {', '.join(self.file_extensions)}")
        return all_files