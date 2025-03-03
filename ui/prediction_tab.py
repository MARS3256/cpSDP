# ui/prediction_tab.py

import os
import pandas as pd
import webbrowser
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                           QFileDialog, QGroupBox, QFormLayout, QLineEdit, QTableView, 
                           QMessageBox, QProgressBar, QSplitter, QComboBox)
from PyQt6.QtCore import Qt, QAbstractTableModel, QModelIndex, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QDesktopServices, QColor
from PyQt6.QtCore import QUrl

from core.predictor import Predictor
from core.utils import generate_report, visualize_metrics

class PredictionTableModel(QAbstractTableModel):
    """Model for displaying prediction results in a table view."""
    
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
        elif role == Qt.ItemDataRole.BackgroundRole:
            # Color rows based on defect prediction
            if 'predicted_defective' in self._data.columns:
                is_defect = self._data.iloc[index.row()]['predicted_defective']
                if is_defect == 1:
                    # Light red background for defective files
                    return QColor(255, 200, 200)
                else:
                    # Light green background for non-defective files
                    return QColor(200, 255, 200)
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return str(self._data.columns[section])
        if orientation == Qt.Orientation.Vertical and role == Qt.ItemDataRole.DisplayRole:
            return str(self._data.index[section])
        return None


class PredictionWorker(QThread):
    """Worker thread for running predictions."""
    progress = pyqtSignal(int)
    finished = pyqtSignal(pd.DataFrame)
    error = pyqtSignal(str)
    
    def __init__(self, model_path, metrics_path):
        """Initialize the prediction worker."""
        super().__init__()
        self.model_path = model_path
        self.metrics_path = metrics_path
    
    def run(self):
        """Run predictions."""
        try:
            # Create predictor
            predictor = Predictor(self.model_path)
            
            # Load metrics
            metrics_df = pd.read_csv(self.metrics_path)
            
            # Make predictions
            result_df = predictor.predict(metrics_df)
            
            self.finished.emit(result_df)
        except Exception as e:
            self.error.emit(str(e))


class PredictionTab(QWidget):
    """Tab for defect prediction."""
    
    def __init__(self, parent=None):
        """Initialize the prediction tab."""
        super().__init__(parent)
        self.parent = parent
        self.model_path = None
        self.metrics_path = None
        self.results_df = None
        self.worker = None
        self.report_path = None
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI components."""
        # Main layout
        main_layout = QVBoxLayout(self)
        
        # Create a splitter for flexible resizing
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Top section - Model and data selection
        top_widget = QWidget()
        top_layout = QVBoxLayout(top_widget)
        self.create_selection_section(top_layout)
        
        # Bottom section - Results display
        bottom_widget = QWidget()
        bottom_layout = QVBoxLayout(bottom_widget)
        self.create_results_section(bottom_layout)
        
        # Add widgets to splitter
        splitter.addWidget(top_widget)
        splitter.addWidget(bottom_widget)
        
        # Add splitter to main layout
        main_layout.addWidget(splitter)
        
        # Set initial splitter sizes
        splitter.setSizes([150, 450])
    
    def create_selection_section(self, parent_layout):
        """Create the model and data selection section."""
        # Model selection group
        model_group = QGroupBox("Model Selection")
        model_layout = QVBoxLayout(model_group)
        
        # Model path selection
        model_path_layout = QHBoxLayout()
        model_path_label = QLabel("Model Path:")
        self.model_path_input = QLineEdit()
        self.model_path_input.setReadOnly(True)
        model_browse_button = QPushButton("Browse...")
        model_browse_button.clicked.connect(self.browse_model)
        
        model_path_layout.addWidget(model_path_label)
        model_path_layout.addWidget(self.model_path_input)
        model_path_layout.addWidget(model_browse_button)
        
        model_layout.addLayout(model_path_layout)
        
        # Data selection group
        data_group = QGroupBox("Data Selection")
        data_layout = QVBoxLayout(data_group)
        
        # Metrics path selection
        metrics_path_layout = QHBoxLayout()
        metrics_path_label = QLabel("Metrics CSV:")
        self.metrics_path_input = QLineEdit()
        self.metrics_path_input.setReadOnly(True)
        metrics_browse_button = QPushButton("Browse...")
        metrics_browse_button.clicked.connect(self.browse_metrics)
        
        metrics_path_layout.addWidget(metrics_path_label)
        metrics_path_layout.addWidget(self.metrics_path_input)
        metrics_path_layout.addWidget(metrics_browse_button)
        
        data_layout.addLayout(metrics_path_layout)
        
        # Create a horizontal layout for the two groups
        selection_layout = QHBoxLayout()
        selection_layout.addWidget(model_group)
        selection_layout.addWidget(data_group)
        
        parent_layout.addLayout(selection_layout)
        
        # Run prediction button
        button_layout = QHBoxLayout()
        
        self.predict_button = QPushButton("Run Prediction")
        self.predict_button.clicked.connect(self.run_prediction)
        self.predict_button.setEnabled(False)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        button_layout.addStretch(1)
        button_layout.addWidget(self.predict_button)
        button_layout.addWidget(self.progress_bar)
        button_layout.addStretch(1)
        
        parent_layout.addLayout(button_layout)
    
    def create_results_section(self, parent_layout):
        """Create the results display section."""
        # Results group
        results_group = QGroupBox("Prediction Results")
        results_layout = QVBoxLayout(results_group)
        
        # Summary stats
        self.summary_label = QLabel("No predictions available")
        results_layout.addWidget(self.summary_label)
        
        # Table view for results
        self.results_table = QTableView()
        self.results_table.setSortingEnabled(True)
        results_layout.addWidget(self.results_table)
        
        # Actions layout
        actions_layout = QHBoxLayout()
        
        # Save results button
        self.save_button = QPushButton("Save Results")
        self.save_button.clicked.connect(self.save_results)
        self.save_button.setEnabled(False)
        
        # Generate report button
        self.report_button = QPushButton("Generate Report")
        self.report_button.clicked.connect(self.generate_report)
        self.report_button.setEnabled(False)
        
        # View report button
        self.view_button = QPushButton("View Report")
        self.view_button.clicked.connect(self.view_report)
        self.view_button.setEnabled(False)
        
        actions_layout.addWidget(self.save_button)
        actions_layout.addWidget(self.report_button)
        actions_layout.addWidget(self.view_button)
        
        results_layout.addLayout(actions_layout)
        
        parent_layout.addWidget(results_group)
    
    def browse_model(self):
        """Open a dialog to select a model file or directory."""
        model_path = QFileDialog.getExistingDirectory(
            self, "Select Model Directory", "", QFileDialog.Option.ShowDirsOnly
        )
        
        if model_path:
            # Check if it's a valid model directory
            model_file = os.path.join(model_path, "model.joblib")
            if not os.path.exists(model_file):
                QMessageBox.warning(
                    self, 
                    "Invalid Model Directory", 
                    "The selected directory does not contain a valid model file (model.joblib)."
                )
                return
            
            self.model_path = model_path
            self.model_path_input.setText(model_path)
            self.update_predict_button()
            
            if self.parent and hasattr(self.parent, 'status_bar'):
                self.parent.status_bar.showMessage(f"Model loaded: {model_path}")
    
    def browse_metrics(self):
        """Open a dialog to select a metrics CSV file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Metrics CSV", "", "CSV Files (*.csv)"
        )
        
        if file_path:
            self.metrics_path = file_path
            self.metrics_path_input.setText(file_path)
            self.update_predict_button()
            
            if self.parent and hasattr(self.parent, 'status_bar'):
                self.parent.status_bar.showMessage(f"Metrics loaded: {file_path}")
    
    def update_predict_button(self):
        """Enable or disable the predict button based on selections."""
        self.predict_button.setEnabled(
            self.model_path is not None and 
            self.metrics_path is not None
        )
    
    def run_prediction(self):
        """Run defect prediction using the selected model and data."""
        if not self.model_path or not self.metrics_path:
            QMessageBox.warning(
                self, 
                "Missing Input", 
                "Please select both a model and a metrics CSV file."
            )
            return
        
        # Update UI
        self.predict_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        
        if self.parent and hasattr(self.parent, 'status_bar'):
            self.parent.status_bar.showMessage("Running prediction...")
        
        # Create worker thread
        self.worker = PredictionWorker(self.model_path, self.metrics_path)
        self.worker.finished.connect(self.on_prediction_finished)
        self.worker.error.connect(self.on_prediction_error)
        self.worker.start()
    
    def on_prediction_finished(self, results_df):
        """Handle completion of prediction."""
        self.results_df = results_df
        self.progress_bar.setVisible(False)
        self.predict_button.setEnabled(True)
        
        # Update the table view
        model = PredictionTableModel(results_df)
        self.results_table.setModel(model)
        
        # Update summary
        defect_count = results_df['predicted_defective'].sum()
        total_count = len(results_df)
        defect_pct = (defect_count / total_count) * 100 if total_count > 0 else 0
        
        summary_text = (
            f"<b>Prediction Results:</b> Found {defect_count} potentially defective files "
            f"out of {total_count} ({defect_pct:.1f}%)."
        )
        self.summary_label.setText(summary_text)
        
        # Enable action buttons
        self.save_button.setEnabled(True)
        self.report_button.setEnabled(True)
        
        # Set report path to None since we haven't generated one yet
        self.report_path = None
        self.view_button.setEnabled(False)
        
        if self.parent and hasattr(self.parent, 'status_bar'):
            self.parent.status_bar.showMessage(
                f"Prediction complete: {defect_count} potential defects found"
            )
    
    def on_prediction_error(self, error_msg):
        """Handle errors during prediction."""
        self.progress_bar.setVisible(False)
        self.predict_button.setEnabled(True)
        
        QMessageBox.critical(self, "Prediction Error", error_msg)
        
        if self.parent and hasattr(self.parent, 'status_bar'):
            self.parent.status_bar.showMessage("Prediction failed")
    
    def save_results(self):
        """Save the prediction results to a CSV file."""
        if self.results_df is None:
            QMessageBox.warning(self, "No Results", "No prediction results available to save.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Prediction Results", "", "CSV Files (*.csv)"
        )
        
        if file_path:
            try:
                if not file_path.endswith('.csv'):
                    file_path += '.csv'
                
                self.results_df.to_csv(file_path, index=False)
                QMessageBox.information(
                    self, "Success", f"Prediction results saved to {file_path}"
                )
                
                if self.parent and hasattr(self.parent, 'status_bar'):
                    self.parent.status_bar.showMessage(f"Results saved to: {file_path}")
            except Exception as e:
                QMessageBox.critical(
                    self, "Error", f"Failed to save results: {str(e)}"
                )
    
    def generate_report(self):
        """Generate an HTML report of prediction results."""
        if self.results_df is None:
            QMessageBox.warning(self, "No Results", "No prediction results available for report.")
            return
        
        report_dir, _ = QFileDialog.getSaveFileName(
            self, "Save Report", "", "HTML Files (*.html)"
        )
        
        if report_dir:
            try:
                if not report_dir.endswith('.html'):
                    report_dir += '.html'
                
                # Generate the report
                generate_report(
                    self.results_df,
                    self.model_path,
                    self.metrics_path,
                    report_dir
                )
                
                self.report_path = report_dir
                self.view_button.setEnabled(True)
                
                QMessageBox.information(
                    self, "Success", f"Report generated and saved to {report_dir}"
                )
                
                if self.parent and hasattr(self.parent, 'status_bar'):
                    self.parent.status_bar.showMessage(f"Report generated: {report_dir}")
            except Exception as e:
                QMessageBox.critical(
                    self, "Error", f"Failed to generate report: {str(e)}"
                )
    
    def view_report(self):
        """Open the generated report in a web browser."""
        if not self.report_path or not os.path.exists(self.report_path):
            QMessageBox.warning(
                self, "No Report", "No report available. Please generate a report first."
            )
            return
        
        try:
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.report_path))
        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Failed to open report: {str(e)}"
            )
    
    def predict_defects(self):
        """Predict defects (called from parent window)."""
        if self.model_path and self.metrics_path:
            self.run_prediction()
        else:
            QMessageBox.information(
                self,
                "Prediction Setup",
                "Please select a trained model and metrics data to run prediction."
            )