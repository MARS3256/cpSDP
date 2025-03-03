# ui/main_window.py

import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QTabWidget, QVBoxLayout,
                           QWidget, QMenuBar, QMenu, QToolBar, QStatusBar, 
                           QMessageBox, QFileDialog)
from PyQt6.QtCore import Qt, QSettings, QSize, QPoint
from PyQt6.QtGui import QIcon, QAction

from ui.project_tab import ProjectTab, ExtractionWorker
from ui.metrics_tab import MetricsTab
from ui.prediction_tab import PredictionTab

class MainWindow(QMainWindow):
    """Main window for the Software Defect Prediction Tool."""
    
    def __init__(self):
        """Initialize the main window UI."""
        super().__init__()
        
        # Setup window
        self.setWindowTitle("cpSDP - Software Defect Prediction Tool")
        self.setMinimumSize(800, 600)
        
        # Initialize settings
        self.settings = QSettings("cpSDP", "SoftwareDefectPrediction")
        self.load_settings()
        
        # Create central widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        
        # Create tab widget
        self.tabs = QTabWidget()
        self.project_tab = ProjectTab(self)
        self.metrics_tab = MetricsTab(self)
        self.prediction_tab = PredictionTab(self)
        
        # Add tabs
        self.tabs.addTab(self.project_tab, "Project")
        self.tabs.addTab(self.metrics_tab, "Metrics")
        self.tabs.addTab(self.prediction_tab, "Prediction")
        
        # Add tab widget to layout
        self.layout.addWidget(self.tabs)
        
        # Create menu bar
        self.create_menu_bar()
        
        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Show the window
        self.show()
    
    def create_menu_bar(self):
        """Create the menu bar with actions."""
        self.menu_bar = self.menuBar()
        
        # File menu
        file_menu = self.menu_bar.addMenu("File")
        
        # Open project action
        open_action = QAction("Open Project", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_project)
        file_menu.addAction(open_action)
        
        # Save metrics action
        save_metrics_action = QAction("Save Metrics", self)
        save_metrics_action.setShortcut("Ctrl+S")
        save_metrics_action.triggered.connect(self.save_metrics)
        file_menu.addAction(save_metrics_action)
        
        file_menu.addSeparator()
        
        # Exit action
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Tools menu
        tools_menu = self.menu_bar.addMenu("Tools")
        
        # Extract metrics action
        extract_action = QAction("Extract Metrics", self)
        extract_action.triggered.connect(self.extract_metrics)
        tools_menu.addAction(extract_action)
        
        # Train model action
        train_action = QAction("Train Model", self)
        train_action.triggered.connect(self.train_model)
        tools_menu.addAction(train_action)
        
        # Predict defects action
        predict_action = QAction("Predict Defects", self)
        predict_action.triggered.connect(self.predict_defects)
        tools_menu.addAction(predict_action)
        
        # Help menu
        help_menu = self.menu_bar.addMenu("Help")
        
        # About action
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def open_project(self):
        """Open a project directory."""
        project_dir = QFileDialog.getExistingDirectory(
            self, "Select Project Directory", "", QFileDialog.Option.ShowDirsOnly
        )
        if project_dir:
            self.project_tab.set_project_path(project_dir)
            self.status_bar.showMessage(f"Project opened: {project_dir}")
    
    def save_metrics(self):
        """Save metrics data to CSV file."""
        self.metrics_tab.save_metrics()
    
    def extract_metrics(self):
        """Extract metrics from the selected project."""
        if not self.project_path:
            QMessageBox.warning(self, "Warning", "Please select a project directory first.")
            return
        
        output_path = self.output_path.text()
        if not output_path:
            output_path, _ = QFileDialog.getSaveFileName(
                self, "Save Metrics CSV", "", "CSV Files (*.csv)"
            )
            if not output_path:
                return
            
            if not output_path.endswith('.csv'):
                output_path += '.csv'
            self.output_path.setText(output_path)
        
        # Get the selected directories
        if self.structure_group.isChecked():
            selected_dirs = self.get_selected_directories()
        else:
            # Default behavior - use the whole project
            selected_dirs = None
        
        # Get the selected language
        language = self.language_combo.currentText()
        
        # Disable UI elements during extraction
        self.extract_button.setEnabled(False)
        self.status_label.setText("Extracting metrics...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        
        # Create worker thread
        self.worker = ExtractionWorker(language, self.project_path, output_path, selected_dirs)
        self.worker.finished.connect(self.on_extraction_finished)
        self.worker.error.connect(self.on_extraction_error)
        self.worker.start()
    
    def train_model(self):
        """Train a model on the current metrics data."""
        self.metrics_tab.train_model()
    
    def predict_defects(self):
        """Predict defects on the current project."""
        self.prediction_tab.predict_defects()
    
    def show_about(self):
        """Show the about dialog."""
        QMessageBox.about(
            self, 
            "About cpSDP",
            "cpSDP - Software Defect Prediction Tool\n\n"
            "A research project focused on the development and analysis of "
            "Software Defect Prediction (SDP) techniques.\n\n"
            "Supports C/C++, Java, and Python code analysis."
        )
    
    def load_settings(self):
        """Load application settings."""
        size = self.settings.value("window_size", QSize(800, 600))
        pos = self.settings.value("window_position", QPoint(100, 100))
        self.resize(size)
        self.move(pos)
    
    def closeEvent(self, event):
        """Save settings before closing."""
        self.settings.setValue("window_size", self.size())
        self.settings.setValue("window_position", self.pos())
        event.accept()


# If running directly, create the application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec())