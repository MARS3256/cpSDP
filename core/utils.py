# core/utils.py

import os
import json
import pandas as pd
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path):
    """
    Load configuration from a JSON file.
    
    Args:
        config_path (str): Path to the config file
        
    Returns:
        dict: Configuration settings
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        return {}

def save_config(config, config_path):
    """
    Save configuration to a JSON file.
    
    Args:
        config (dict): Configuration settings
        config_path (str): Path to save the config file
        
    Returns:
        bool: True if saving was successful
    """
    try:
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        logger.info(f"Saved configuration to {config_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving config: {str(e)}")
        return False

def get_language_extractor(language, project_path):
    """
    Get the appropriate feature extractor for a language.
    
    Args:
        language (str): Programming language ('c/c++', 'java', or 'python')
        project_path (str): Path to the project
        
    Returns:
        FeatureExtractor: The appropriate feature extractor
    """
    from extractors.c_cpp_extractor import CCppExtractor
    from extractors.java_extractor import JavaExtractor
    from extractors.python_extractor import PythonExtractor
    
    language = language.lower()
    if language in ['c', 'c++', 'c/c++']:
        return CCppExtractor(project_path)
    elif language == 'java':
        return JavaExtractor(project_path)
    elif language == 'python':
        return PythonExtractor(project_path)
    else:
        logger.error(f"Unsupported language: {language}")
        raise ValueError(f"Unsupported language: {language}")

def visualize_metrics(csv_path, output_dir):
    """
    Create visualizations of metrics from a CSV file.
    
    Args:
        csv_path (str): Path to the CSV file with metrics
        output_dir (str): Directory to save the visualizations
        
    Returns:
        list: Paths to the saved visualization files
    """
    try:
        # Load the CSV data
        data = pd.read_csv(csv_path)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # List of visualization files
        viz_files = []
        
        # Get numeric columns (metrics)
        numeric_cols = [col for col in data.columns if col not in ['file_path'] and pd.api.types.is_numeric_dtype(data[col])]
        
        # 1. Distribution of metrics
        plt.figure(figsize=(15, 10))
        for i, col in enumerate(numeric_cols[:min(9, len(numeric_cols))]):  # Up to 9 metrics
            plt.subplot(3, 3, i+1)
            sns.histplot(data[col], kde=True)
            plt.title(f'Distribution of {col}')
        plt.tight_layout()
        
        dist_file = os.path.join(output_dir, 'metric_distributions.png')
        plt.savefig(dist_file)
        plt.close()
        viz_files.append(dist_file)
        
        # 2. Correlation heatmap
        plt.figure(figsize=(12, 10))
        corr = data[numeric_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, linewidths=.5)
        plt.title('Correlation Matrix of Metrics')
        
        corr_file = os.path.join(output_dir, 'correlation_heatmap.png')
        plt.savefig(corr_file)
        plt.close()
        viz_files.append(corr_file)
        
        # 3. If defective column exists, analyze it
        if 'defective' in data.columns:
            plt.figure(figsize=(10, 8))
            
            # Distribution of defective vs non-defective
            plt.subplot(2, 2, 1)
            sns.countplot(x='defective', data=data)
            plt.title('Distribution of Defective Files')
            
            # Top 3 metrics most correlated with defects
            defect_corr = data[numeric_cols].corr()['defective'].abs().sort_values(ascending=False)
            top_metrics = defect_corr.index[1:4]  # Skip 'defective' itself
            
            for i, metric in enumerate(top_metrics):
                plt.subplot(2, 2, i+2)
                sns.boxplot(x='defective', y=metric, data=data)
                plt.title(f'{metric} by Defect Status')
            
            plt.tight_layout()
            
            defect_file = os.path.join(output_dir, 'defect_analysis.png')
            plt.savefig(defect_file)
            plt.close()
            viz_files.append(defect_file)
        
        # 4. If predicted_defective column exists, analyze it
        if 'predicted_defective' in data.columns and 'defective' in data.columns:
            plt.figure(figsize=(10, 8))
            
            # Confusion matrix visualization
            conf_matrix = pd.crosstab(data['defective'], data['predicted_defective'], 
                                      rownames=['Actual'], colnames=['Predicted'])
            
            plt.subplot(2, 2, 1)
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            
            # ROC curve if probability available
            if 'defect_probability' in data.columns:
                from sklearn.metrics import roc_curve, auc
                fpr, tpr, _ = roc_curve(data['defective'], data['defect_probability'])
                roc_auc = auc(fpr, tpr)
                
                plt.subplot(2, 2, 2)
                plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC Curve')
                plt.legend(loc='lower right')
            
            plt.tight_layout()
            
            pred_file = os.path.join(output_dir, 'prediction_analysis.png')
            plt.savefig(pred_file)
            plt.close()
            viz_files.append(pred_file)
        
        logger.info(f"Created {len(viz_files)} visualization files in {output_dir}")
        
        return viz_files
    
    except Exception as e:
        logger.error(f"Error creating visualizations: {str(e)}")
        return []
def generate_report(data_path, output_path, title="Software Defect Prediction Report"):
    """
    Generate an HTML report from prediction results.
    
    Args:
        data_path (str): Path to CSV with prediction results
        output_path (str): Path to save the HTML report
        title (str): Title of the report
        
    Returns:
        str: Path to the saved HTML report
    """
    try:
        # Load data
        data = pd.read_csv(data_path)
        
        # Create report directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Basic stats
        total_files = len(data)
        
        if 'predicted_defective' in data.columns:
            pred_defective = sum(data['predicted_defective'])
            pred_percentage = (pred_defective / total_files) * 100 if total_files > 0 else 0
        
        if 'defective' in data.columns and 'predicted_defective' in data.columns:
            true_defective = sum(data['defective'])
            true_percentage = (true_defective / total_files) * 100 if total_files > 0 else 0
            
            # Calculate metrics
            true_pos = sum((data['defective'] == 1) & (data['predicted_defective'] == 1))
            false_pos = sum((data['defective'] == 0) & (data['predicted_defective'] == 1))
            false_neg = sum((data['defective'] == 1) & (data['predicted_defective'] == 0))
            true_neg = sum((data['defective'] == 0) & (data['predicted_defective'] == 0))
            
            # Calculate performance metrics
            accuracy = (true_pos + true_neg) / total_files if total_files > 0 else 0
            precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
            recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Generate HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    line-height: 1.6;
                }}
                h1, h2, h3 {{
                    color: #2c3e50;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin-bottom: 20px;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
                tr:nth-child(even) {{
                    background-color: #f2f2f2;
                }}
                .metrics-container {{
                    display: flex;
                    flex-wrap: wrap;
                    justify-content: space-between;
                }}
                .metric-box {{
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    padding: 15px;
                    margin: 10px;
                    flex: 1;
                    min-width: 200px;
                    background-color: #f9f9f9;
                }}
                .visualization {{
                    margin: 20px 0;
                    text-align: center;
                }}
                .footer {{
                    margin-top: 30px;
                    font-size: 0.8em;
                    color: #7f8c8d;
                    text-align: center;
                }}
            </style>
        </head>
        <body>
            <h1>{title}</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Summary</h2>
            <div class="metrics-container">
                <div class="metric-box">
                    <h3>File Statistics</h3>
                    <p>Total files analyzed: {total_files}</p>
        """
        
        # Add prediction statistics if available
        if 'predicted_defective' in data.columns:
            html_content += f"""
                    <p>Files predicted as defective: {pred_defective} ({pred_percentage:.1f}%)</p>
            """
        
        # Add ground truth statistics if available
        if 'defective' in data.columns:
            html_content += f"""
                    <p>Actually defective files: {true_defective} ({true_percentage:.1f}%)</p>
                </div>
            """
        else:
            html_content += """
                </div>
            """
        
        # Add performance metrics if both actual and predicted data are available
        if 'defective' in data.columns and 'predicted_defective' in data.columns:
            html_content += f"""
                <div class="metric-box">
                    <h3>Performance Metrics</h3>
                    <p>Accuracy: {accuracy:.4f}</p>
                    <p>Precision: {precision:.4f}</p>
                    <p>Recall: {recall:.4f}</p>
                    <p>F1 Score: {f1_score:.4f}</p>
                </div>
                
                <div class="metric-box">
                    <h3>Confusion Matrix</h3>
                    <p>True Positives: {true_pos}</p>
                    <p>False Positives: {false_pos}</p>
                    <p>True Negatives: {true_neg}</p>
                    <p>False Negatives: {false_neg}</p>
                </div>
            </div>
            """
        else:
            html_content += """
            </div>
            """
        
        # Add visualizations if they exist
        viz_dir = os.path.join(os.path.dirname(output_path), 'visualizations')
        if os.path.exists(viz_dir):
            html_content += """
            <h2>Visualizations</h2>
            """
            for viz_file in sorted(os.listdir(viz_dir)):
                if viz_file.endswith(('.png', '.jpg', '.svg')):
                    rel_path = os.path.join('visualizations', viz_file)
                    viz_name = ' '.join(os.path.splitext(viz_file)[0].split('_')).title()
                    html_content += f"""
                    <div class="visualization">
                        <h3>{viz_name}</h3>
                        <img src="{rel_path}" alt="{viz_name}" style="max-width: 100%;">
                    </div>
                    """
        
        # Add top potentially defective files
        if 'predicted_defective' in data.columns and 'defect_probability' in data.columns:
            html_content += """
            <h2>Top Potentially Defective Files</h2>
            <table>
                <tr>
                    <th>File</th>
                    <th>Defect Probability</th>
                    <th>Predicted</th>
                </tr>
            """
            
            # Sort by defect probability and take top 20
            top_files = data.sort_values('defect_probability', ascending=False).head(20)
            for _, row in top_files.iterrows():
                file_path = row['file_path']
                probability = row['defect_probability']
                predicted = "Yes" if row['predicted_defective'] == 1 else "No"
                html_content += f"""
                <tr>
                    <td>{file_path}</td>
                    <td>{probability:.4f}</td>
                    <td>{predicted}</td>
                </tr>
                """
            
            html_content += """
            </table>
            """
        
        # Close the HTML document
        html_content += """
            <div class="footer">
                <p>Generated by cpSDP - Software Defect Prediction Tool</p>
            </div>
        </body>
        </html>
        """
        
        # Write the HTML content to file
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Report generated at {output_path}")
        return output_path
    
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        return None


def normalize_metrics(data):
    """
    Normalize metrics data using min-max scaling.
    
    Args:
        data (pandas.DataFrame): DataFrame containing metrics
        
    Returns:
        pandas.DataFrame: Normalized metrics
    """
    try:
        # Get numeric columns (metrics)
        numeric_cols = [col for col in data.columns if col not in ['file_path', 'defective'] 
                        and pd.api.types.is_numeric_dtype(data[col])]
        
        # Create a copy of the data
        normalized = data.copy()
        
        # Apply min-max normalization to each numeric column
        for col in numeric_cols:
            min_val = normalized[col].min()
            max_val = normalized[col].max()
            if max_val > min_val:  # Avoid division by zero
                normalized[col] = (normalized[col] - min_val) / (max_val - min_val)
            else:
                normalized[col] = 0  # If all values are the same
        
        logger.info(f"Normalized {len(numeric_cols)} metric columns")
        return normalized
    
    except Exception as e:
        logger.error(f"Error normalizing metrics: {str(e)}")
        raise


def evaluate_model(true_labels, predictions, probabilities=None):
    """
    Evaluate a defect prediction model with various metrics.
    
    Args:
        true_labels (array-like): True binary labels
        predictions (array-like): Predicted binary labels
        probabilities (array-like, optional): Predicted probabilities for the positive class
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix, matthews_corrcoef
    )
    
    try:
        metrics = {
            'accuracy': accuracy_score(true_labels, predictions),
            'precision': precision_score(true_labels, predictions, zero_division=0),
            'recall': recall_score(true_labels, predictions, zero_division=0),
            'f1_score': f1_score(true_labels, predictions, zero_division=0),
            'mcc': matthews_corrcoef(true_labels, predictions)
        }
        
        if probabilities is not None:
            metrics['auc_roc'] = roc_auc_score(true_labels, probabilities)
        
        # Calculate confusion matrix values
        tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()
        metrics.update({
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn)
        })
        
        return metrics
    
    except Exception as e:
        logger.error(f"Error evaluating model: {str(e)}")
        raise


def feature_importance_analysis(model, feature_names):
    """
    Analyze and return feature importance from a trained model.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names (list): List of feature names
        
    Returns:
        pandas.DataFrame: DataFrame with features and their importance scores
    """
    try:
        if not hasattr(model, 'feature_importances_'):
            logger.warning("Model does not have feature_importances_ attribute")
            return None
        
        # Create DataFrame with feature importances
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        logger.info(f"Extracted importance for {len(feature_names)} features")
        return importance_df
    
    except Exception as e:
        logger.error(f"Error analyzing feature importance: {str(e)}")
        return None


def import_numpy():
    """
    Import numpy and return the module.
    This is needed because there's a reference to np in the code but it's not imported.
    
    Returns:
        module: The numpy module
    """
    import numpy as np
    return np

# Make numpy available as np
np = import_numpy()                                                        
