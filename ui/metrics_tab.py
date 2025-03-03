# ui/metrics_tab.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, 
                           QPushButton, QFileDialog, QTableView, QGroupBox,
                           QSplitter, QMessageBox, QTabWidget, QFormLayout,
                           QSpinBox, QDoubleSpinBox, QCheckBox, QProgressBar)
from PyQt6.QtCore import Qt, QAbstractTableModel, QModelIndex, QThread, pyqtSignal
from PyQt6.QtGui import QFont

from core.utils import visualize_metrics, normalize_metrics, feature_importance_analysis
from core.model_trainer import ModelTrainer

class PandasModel(QAbstractTableModel):
    """Model for displaying pandas DataFrame in a table view."""
    
    def __init__(self, data):
        """Initialize with pandas DataFrame."""
        super().__init__()
        self._data = data

    def rowCount(self, parent=QModelIndex()):
        return self._data.shape[0]

    def columnCount(self, parent=QModelIndex()):
        return self._data.shape[1]

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole:
            value = self._data.iloc[index.row(), index.column()]
            if isinstance(value, (int, float)):
                # Format numbers nicely
                return f"{value:g}" if value == value else ""  # handle NaN
            return str(value)
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return str(self._data.columns[section])
        if orientation == Qt.Orientation.Vertical and role == Qt.ItemDataRole.DisplayRole:
            return str(self._data.index[section])
        return None


class TrainingWorker(QThread):
    """Worker thread for model training."""
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict, str)
    error = pyqtSignal(str)
    
    def __init__(self, csv_path, output_dir, params):
        """Initialize the training worker."""
        super().__init__()
        self.csv_path = csv_path
        self.output_dir = output_dir
        self.params = params
    
    def run(self):
        """Train the model."""
        try:
            # Create trainer
            trainer = ModelTrainer(self.csv_path)
            
            # Load data
            trainer.load_data()
            
            # Train model with parameters
            model, metrics = trainer.train_model(
                n_estimators=self.params.get('n_estimators', 100),
                max_depth=self.params.get('max_depth', None)
            )
            
            # Run cross-validation if requested
            if self.params.get('do_cv', False):
                cv_metrics = trainer.cross_validate(cv=self.params.get('cv_folds', 5))
                metrics['cross_validation'] = cv_metrics
            
            # Save the model
            os.makedirs(self.output_dir, exist_ok=True)
            model_path, _, _ = trainer.save_model(self.output_dir)
            
            self.finished.emit(metrics, model_path)
        except Exception as e:
            self.error.emit(str(e))


class MetricsTab(QWidget):
    """Tab for metrics visualization and model training."""
    
    def __init__(self, parent=None):
        """Initialize the metrics tab."""
        super().__init__(parent)
        self.parent = parent
        self.metrics_df = None
        self.csv_path = None
        self.model_path = None
        self.worker = None
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI components."""
        # Main layout
        main_layout = QVBoxLayout(self)
        
        # Create a splitter for flexible resizing
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Top section - Data loading and basic metrics
        top_widget = QWidget()
        top_layout = QVBoxLayout(top_widget)
        
        # Metrics data section
        self.create_data_section(top_layout)
        
        # Bottom section - Tabs for visualizations and model training
        bottom_widget = QWidget()
        bottom_layout = QVBoxLayout(bottom_widget)
        
        # Create tabs for different features
        tabs = QTabWidget()
        
        # Visualization tab
        viz_tab = QWidget()
        self.create_visualization_tab(viz_tab)
        tabs.addTab(viz_tab, "Visualizations")
        
        # Model training tab
        train_tab = QWidget()
        self.create_training_tab(train_tab)
        tabs.addTab(train_tab, "Model Training")
        
        # Feature importance tab
        feature_tab = QWidget()
        self.create_feature_importance_tab(feature_tab)
        tabs.addTab(feature_tab, "Feature Importance")
        
        bottom_layout.addWidget(tabs)
        
        # Add widgets to splitter
        splitter.addWidget(top_widget)
        splitter.addWidget(bottom_widget)
        
        # Add splitter to main layout
        main_layout.addWidget(splitter)
        
        # Set initial splitter sizes
        splitter.setSizes([200, 400])
    
    def create_data_section(self, parent_layout):
        """Create the data loading and table view section."""
        # Group box for data loading
        data_group = QGroupBox("Metrics Data")
        data_layout = QVBoxLayout(data_group)
        
        # Controls for loading data
        controls_layout = QHBoxLayout()
        
        load_button = QPushButton("Load Metrics CSV")
        load_button.clicked.connect(self.load_metrics_dialog)
        
        save_button = QPushButton("Save Metrics CSV")
        save_button.clicked.connect(self.save_metrics)
        
        self.filter_combo = QComboBox()
        self.filter_combo.addItem("All Files")
        self.filter_combo.addItem("Defective Files Only")
        self.filter_combo.addItem("Non-Defective Files Only")
        self.filter_combo.currentIndexChanged.connect(self.filter_data)
        
        controls_layout.addWidget(load_button)
        controls_layout.addWidget(save_button)
        controls_layout.addWidget(QLabel("Filter:"))
        controls_layout.addWidget(self.filter_combo)
        
        data_layout.addLayout(controls_layout)
        
        # Status label
        self.data_status = QLabel("No data loaded")
        data_layout.addWidget(self.data_status)
        
        # Table view for data
        self.table_view = QTableView()
        self.table_view.setSortingEnabled(True)
        data_layout.addWidget(self.table_view)
        
        parent_layout.addWidget(data_group)
    
    def create_visualization_tab(self, parent):
        """Create the visualization tab."""
        layout = QVBoxLayout(parent)
        
        # Controls for visualizations
        controls_layout = QHBoxLayout()
        
        self.viz_type_combo = QComboBox()
        self.viz_type_combo.addItems([
            "Metric Distributions", 
            "Correlation Matrix", 
            "Metrics by Defect Status"
        ])
        
        update_button = QPushButton("Update Visualization")
        update_button.clicked.connect(self.update_visualization)
        
        controls_layout.addWidget(QLabel("Visualization Type:"))
        controls_layout.addWidget(self.viz_type_combo)
        controls_layout.addWidget(update_button)
        controls_layout.addStretch(1)
        
        layout.addLayout(controls_layout)
        
        # Canvas for visualizations
        self.figure = Figure(figsize=(8, 6), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
    
    def create_training_tab(self, parent):
        """Create the model training tab."""
        layout = QVBoxLayout(parent)
        
        # Training parameters group
        params_group = QGroupBox("Training Parameters")
        params_layout = QFormLayout(params_group)
        
        self.n_estimators_spin = QSpinBox()
        self.n_estimators_spin.setRange(10, 1000)
        self.n_estimators_spin.setValue(100)
        self.n_estimators_spin.setSingleStep(10)
        params_layout.addRow("Number of Trees:", self.n_estimators_spin)
        
        self.max_depth_spin = QSpinBox()
        self.max_depth_spin.setRange(0, 100)
        self.max_depth_spin.setValue(0)
        self.max_depth_spin.setSpecialValueText("None (Unlimited)")
        params_layout.addRow("Max Tree Depth:", self.max_depth_spin)
        
        self.test_split_spin = QDoubleSpinBox()
        self.test_split_spin.setRange(0.1, 0.5)
        self.test_split_spin.setValue(0.2)
        self.test_split_spin.setSingleStep(0.05)
        params_layout.addRow("Test Split:", self.test_split_spin)
        
        self.cv_check = QCheckBox("Perform Cross-Validation")
        self.cv_check.setChecked(True)
        params_layout.addRow("", self.cv_check)
        
        self.cv_folds_spin = QSpinBox()
        self.cv_folds_spin.setRange(2, 10)
        self.cv_folds_spin.setValue(5)
        params_layout.addRow("CV Folds:", self.cv_folds_spin)
        
        layout.addWidget(params_group)
        
        # Training controls
        controls_layout = QHBoxLayout()
        
        self.train_button = QPushButton("Train Model")
        self.train_button.clicked.connect(self.train_model)
        self.train_button.setEnabled(False)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        self.model_status = QLabel("No model trained")
        
        controls_layout.addWidget(self.train_button)
        controls_layout.addWidget(self.progress_bar)
        controls_layout.addWidget(self.model_status)
        
        layout.addLayout(controls_layout)
        
        # Model metrics group
        metrics_group = QGroupBox("Model Metrics")
        metrics_layout = QVBoxLayout(metrics_group)
        
        self.metrics_text = QLabel("No metrics available")
        self.metrics_text.setAlignment(Qt.AlignmentFlag.AlignTop)
        metrics_layout.addWidget(self.metrics_text)
        
        layout.addWidget(metrics_group)
        layout.setStretchFactor(metrics_group, 1)
    
    def create_feature_importance_tab(self, parent):
        """Create the feature importance tab."""
        layout = QVBoxLayout(parent)
        
        # Information label
        info_label = QLabel("Feature importance will be shown after a model is trained.")
        layout.addWidget(info_label)
        
        # Canvas for feature importance visualization
        self.feature_figure = Figure(figsize=(8, 6), tight_layout=True)
        self.feature_canvas = FigureCanvas(self.feature_figure)
        layout.addWidget(self.feature_canvas)
    
    def load_metrics_dialog(self):
        """Open a dialog to select a metrics CSV file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Metrics CSV", "", "CSV Files (*.csv)"
        )
        
        if file_path:
            self.load_metrics(file_path)
    
    def load_metrics(self, csv_path):
        """Load metrics from a CSV file."""
        try:
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"CSV file not found: {csv_path}")
            
            self.metrics_df = pd.read_csv(csv_path)
            self.csv_path = csv_path
            
            # Make sure the metrics dataframe has all required columns
            if 'file_path' not in self.metrics_df.columns:
                self.metrics_df['file_path'] = [f"file_{i}" for i in range(len(self.metrics_df))]
            
            if 'defective' not in self.metrics_df.columns:
                self.metrics_df['defective'] = 0
            
            # Update the table view
            model = PandasModel(self.metrics_df)
            self.table_view.setModel(model)
            
            # Update status
            file_name = os.path.basename(csv_path)
            defective_count = self.metrics_df['defective'].sum()
            total_count = len(self.metrics_df)
            self.data_status.setText(
                f"Loaded {file_name} with {total_count} files "
                f"({defective_count} defective, {total_count - defective_count} non-defective)"
            )
            
            # Enable model training
            self.train_button.setEnabled(True)
            
            # Update visualization
            self.update_visualization()
            
            # Notify the parent window
            if self.parent and hasattr(self.parent, 'status_bar'):
                self.parent.status_bar.showMessage(f"Metrics loaded from: {csv_path}")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load metrics: {str(e)}")
            if self.parent and hasattr(self.parent, 'status_bar'):
                self.parent.status_bar.showMessage("Error loading metrics")
    
    def save_metrics(self):
        """Save the current metrics to a CSV file."""
        if self.metrics_df is None:
            QMessageBox.warning(self, "Warning", "No metrics data to save.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Metrics CSV", "", "CSV Files (*.csv)"
        )
        
        if file_path:
            try:
                if not file_path.endswith('.csv'):
                    file_path += '.csv'
                
                self.metrics_df.to_csv(file_path, index=False)
                QMessageBox.information(
                    self, "Success", f"Metrics saved to {file_path}"
                )
                
                if self.parent and hasattr(self.parent, 'status_bar'):
                    self.parent.status_bar.showMessage(f"Metrics saved to: {file_path}")
            
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save metrics: {str(e)}")
                if self.parent and hasattr(self.parent, 'status_bar'):
                    self.parent.status_bar.showMessage("Error saving metrics")
    
    def filter_data(self):
        """Filter the data based on the selected filter."""
        if self.metrics_df is None:
            return
        
        filter_idx = self.filter_combo.currentIndex()
        filtered_df = None
        
        if filter_idx == 0:  # All Files
            filtered_df = self.metrics_df
        elif filter_idx == 1:  # Defective Files Only
            filtered_df = self.metrics_df[self.metrics_df['defective'] == 1]
        elif filter_idx == 2:  # Non-Defective Files Only
            filtered_df = self.metrics_df[self.metrics_df['defective'] == 0]
        
        model = PandasModel(filtered_df)
        self.table_view.setModel(model)
    
    def update_visualization(self):
        """Update the visualization based on the selected type."""
        if self.metrics_df is None:
            return
        
        viz_type = self.viz_type_combo.currentText()
        self.figure.clear()
        
        if viz_type == "Metric Distributions":
            self.visualize_distributions()
        elif viz_type == "Correlation Matrix":
            self.visualize_correlation()
        elif viz_type == "Metrics by Defect Status":
            self.visualize_defect_metrics()
        
        self.canvas.draw()
    
    def visualize_distributions(self):
        """Visualize the distributions of metrics."""
        # Get numeric columns (excluding file_path and defective)
        numeric_cols = [col for col in self.metrics_df.columns 
                        if col not in ['file_path', 'defective'] 
                        and pd.api.types.is_numeric_dtype(self.metrics_df[col])]
        
        # Limit to 9 metrics for readability
        plot_cols = numeric_cols[:min(9, len(numeric_cols))]
        rows = (len(plot_cols) + 2) // 3  # Ceil division
        
        for i, col in enumerate(plot_cols):
            ax = self.figure.add_subplot(rows, 3, i+1)
            self.metrics_df[col].hist(bins=20, ax=ax)
            ax.set_title(col)
        
        self.figure.suptitle("Metric Distributions", fontsize=16)
        self.figure.tight_layout()
    
    def visualize_correlation(self):
        """Visualize the correlation matrix of metrics."""
        # Get numeric columns (excluding file_path)
        numeric_cols = [col for col in self.metrics_df.columns 
                        if col != 'file_path' 
                        and pd.api.types.is_numeric_dtype(self.metrics_df[col])]
        
        # Calculate correlation matrix
        corr = self.metrics_df[numeric_cols].corr()
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        # Create heatmap
        ax = self.figure.add_subplot(111)
        im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
        
        # Add colorbar
        self.figure.colorbar(im, ax=ax, shrink=0.8)
        
        # Add labels
        ax.set_xticks(np.arange(len(numeric_cols)))
        ax.set_yticks(np.arange(len(numeric_cols)))
        ax.set_xticklabels(numeric_cols, rotation=45, ha='right')
        ax.set_yticklabels(numeric_cols)
        
        # Loop over data dimensions and create text annotations
        for i in range(len(numeric_cols)):
            for j in range(len(numeric_cols)):
                if i > j:  # Lower triangle only
                    ax.text(j, i, f"{corr.iloc[i, j]:.2f}",
                           ha="center", va="center", 
                           color="white" if abs(corr.iloc[i, j]) > 0.5 else "black")
        
        ax.set_title("Correlation Matrix of Metrics")
        self.figure.tight_layout()
    
    def visualize_defect_metrics(self):
        """Visualize metrics by defect status."""
        if 'defective' not in self.metrics_df.columns:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, "No defect data available", 
                   horizontalalignment='center', verticalalignment='center')
            return
        
        # Count defective vs non-defective
        defect_counts = self.metrics_df['defective'].value_counts()
        
        # Create subplots
        ax1 = self.figure.add_subplot(221)
        ax1.bar(['Non-Defective', 'Defective'], 
               [defect_counts.get(0, 0), defect_counts.get(1, 0)])
        ax1.set_title('Defect Distribution')
        
        # Show top metrics with highest correlation to defect status
        # Get numeric columns (excluding file_path and defective)
        numeric_cols = [col for col in self.metrics_df.columns 
                        if col not in ['file_path', 'defective'] 
                        and pd.api.types.is_numeric_dtype(self.metrics_df[col])]
        
        # Calculate correlation with defective column
        corr_with_defect = []
        for col in numeric_cols:
            corr = self.metrics_df[['defective', col]].corr().iloc[0, 1]
            corr_with_defect.append((col, abs(corr)))
        
        # Sort by absolute correlation value
        corr_with_defect.sort(key=lambda x: x[1], reverse=True)
        
        # Take top 3 metrics
        top_metrics = [item[0] for item in corr_with_defect[:3]]
        
        # Plot boxplots for top metrics by defect status
        for i, metric in enumerate(top_metrics):
            ax = self.figure.add_subplot(2, 2, i+2)
            data = [
                self.metrics_df[self.metrics_df['defective'] == 0][metric],
                self.metrics_df[self.metrics_df['defective'] == 1][metric]
            ]
            ax.boxplot(data, labels=['Non-Defective', 'Defective'])
            ax.set_title(f'{metric} by Defect Status')
        
        self.figure.tight_layout()
    
    def train_model(self):
        """Train a defect prediction model."""
        if self.metrics_df is None:
            QMessageBox.warning(self, "Warning", "No metrics data loaded.")
            return
        
        if 'defective' not in self.metrics_df.columns:
            QMessageBox.warning(self, "Warning", "Metrics data does not contain 'defective' column.")
            return
        
        # Check if defective column has any variation (both 0 and 1 values)
        defective_vals = self.metrics_df['defective'].unique()
        if len(defective_vals) < 2:
            QMessageBox.warning(
                self, 
                "Warning", 
                "The 'defective' column doesn't have both positive and negative examples. "
                "Please ensure the data contains both defective and non-defective files."
            )
            return
        
        # Get output directory for model
        output_dir, _ = QFileDialog.getSaveFileName(
            self, "Save Model", "", "Model Directory"
        )
        
        if not output_dir:
            return
        
        # Get training parameters
        params = {
            'n_estimators': self.n_estimators_spin.value(),
            'max_depth': self.max_depth_spin.value() if self.max_depth_spin.value() > 0 else None,
            'test_size': self.test_split_spin.value(),
            'do_cv': self.cv_check.isChecked(),
            'cv_folds': self.cv_folds_spin.value()
        }
        
        # Update UI
        self.train_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.model_status.setText("Training model...")
        
        # Create and start worker thread
        self.worker = TrainingWorker(self.csv_path, output_dir, params)
        self.worker.finished.connect(self.on_training_finished)
        self.worker.error.connect(self.on_training_error)
        self.worker.start()
    
    def on_training_finished(self, metrics, model_path):
        """Handle completion of model training."""
        self.progress_bar.setVisible(False)
        self.train_button.setEnabled(True)
        self.model_path = model_path
        self.model_status.setText(f"Model saved to: {model_path}")
        
        # Update metrics display
        metrics_text = f"<b>Model Performance:</b><br>"
        metrics_text += f"Accuracy: {metrics['accuracy']:.4f}<br>"
        metrics_text += f"Precision: {metrics['precision']:.4f}<br>"
        metrics_text += f"Recall: {metrics['recall']:.4f}<br>"
        metrics_text += f"F1 Score: {metrics['f1_score']:.4f}<br>"
        
        if 'cross_validation' in metrics:
            cv = metrics['cross_validation']
            metrics_text += "<br><b>Cross-Validation Results (mean ± std):</b><br>"
            metrics_text += f"Accuracy: {cv['accuracy']['mean']:.4f} ± {cv['accuracy']['std']:.4f}<br>"
            metrics_text += f"Precision: {cv['precision']['mean']:.4f} ± {cv['precision']['std']:.4f}<br>"
            metrics_text += f"Recall: {cv['recall']['mean']:.4f} ± {cv['recall']['std']:.4f}<br>"
            metrics_text += f"F1 Score: {cv['f1_score']['mean']:.4f} ± {cv['f1_score']['std']:.4f}<br>"
        
        self.metrics_text.setText(metrics_text)
        
        # Update feature importance visualization
        self.visualize_feature_importance()
        
        # Show success message
        QMessageBox.information(
            self,
            "Training Complete",
            f"Model has been successfully trained and saved to:\n{model_path}"
        )
        
        # Update parent window status
        if self.parent and hasattr(self.parent, 'status_bar'):
            self.parent.status_bar.showMessage(f"Model trained: {model_path}")
    
    def on_training_error(self, error_msg):
        """Handle errors during model training."""
        self.progress_bar.setVisible(False)
        self.train_button.setEnabled(True)
        self.model_status.setText("Training failed")
        
        QMessageBox.critical(self, "Training Error", error_msg)
        
        if self.parent and hasattr(self.parent, 'status_bar'):
            self.parent.status_bar.showMessage("Model training failed")
    
    def visualize_feature_importance(self):
        """Visualize feature importances."""
        if self.model_path is None:
            return
        
        try:
            # Load the model
            import joblib
            import os
            model = joblib.load(os.path.join(os.path.dirname(self.model_path), 'model.joblib'))
            features = joblib.load(os.path.join(os.path.dirname(self.model_path), 'features.joblib'))
            
            # Get feature importances
            importance_df = feature_importance_analysis(model, features)
            if importance_df is None:
                return
            
            # Sort by importance
            importance_df = importance_df.sort_values('importance', ascending=False)
            
            # Clear the figure
            self.feature_figure.clear()
            
            # Create barplot
            ax = self.feature_figure.add_subplot(111)
            ax.barh(importance_df['feature'][:15], importance_df['importance'][:15])
            ax.set_xlabel('Importance')
            ax.set_title('Feature Importance')
            
            # Invert the y-axis to have the most important feature at the top
            ax.invert_yaxis()
            
            self.feature_canvas.draw()
        
        except Exception as e:
            print(f"Error visualizing feature importance: {str(e)}")