# ui/project_tab.py

import os
import sys
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                           QLineEdit, QPushButton, QComboBox, QFileDialog,
                           QGroupBox, QFormLayout, QTextEdit, QMessageBox,
                           QProgressBar, QTreeWidget, QTreeWidgetItem, QCheckBox, 
                           QHeaderView, QFrame)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont

from core.utils import get_language_extractor

class ExtractionWorker(QThread):
    """Worker thread for extracting metrics from a project."""
    progress = pyqtSignal(int)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    
    def __init__(self, language, project_path, output_path, selected_dirs=None):
        """Initialize the extraction worker."""
        super().__init__()
        self.language = language
        self.project_path = project_path
        self.output_path = output_path
        self.selected_dirs = selected_dirs
    
    def run(self):
        """Extract metrics from the project."""
        try:
            # Get appropriate extractor
            extractor = get_language_extractor(self.language, self.project_path)
            
            # Set selected directories if provided
            if self.selected_dirs:
                extractor.set_selected_directories(self.selected_dirs)
            
            # Extract metrics and save to CSV
            csv_path = extractor.save_metrics_to_csv(self.output_path)
            
            if csv_path:
                self.finished.emit(csv_path)
            else:
                self.error.emit("Failed to save metrics to CSV file.")
        except Exception as e:
            self.error.emit(f"Error during extraction: {str(e)}")


class ProjectTab(QWidget):
    """Tab for project selection and metrics extraction."""
    
    def __init__(self, parent=None):
        """Initialize the project tab."""
        super().__init__(parent)
        self.parent = parent
        self.project_path = ""
        self.metrics_path = ""
        self.extractor = None
        self.worker = None
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI components."""
        # Main layout
        main_layout = QVBoxLayout(self)
        
        # Project selection section
        self.create_project_selection_group(main_layout)
        
        # Project info section
        self.create_project_info_group(main_layout)
        
        # Extraction section
        self.create_extraction_group(main_layout)
        
        # Status section
        self.create_status_section(main_layout)
        
        # Set stretch factors
        main_layout.setStretch(0, 0)  # Project selection
        main_layout.setStretch(1, 1)  # Project info
        main_layout.setStretch(2, 0)  # Extraction
        main_layout.setStretch(3, 0)  # Status
    
    def create_project_selection_group(self, parent_layout):
        """Create the project selection group."""
        group_box = QGroupBox("Project Selection")
        layout = QVBoxLayout(group_box)
        
        # Project path selection
        path_layout = QHBoxLayout()
        path_label = QLabel("Project Path:")
        self.path_input = QLineEdit()
        self.path_input.setReadOnly(True)
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self.browse_project)
        
        path_layout.addWidget(path_label)
        path_layout.addWidget(self.path_input)
        path_layout.addWidget(browse_button)
        
        # Language selection
        language_layout = QHBoxLayout()
        language_label = QLabel("Language:")
        self.language_combo = QComboBox()
        self.language_combo.addItems(["C/C++", "Java", "Python"])
        
        language_layout.addWidget(language_label)
        language_layout.addWidget(self.language_combo)
        language_layout.addStretch(1)
        
        layout.addLayout(path_layout)
        layout.addLayout(language_layout)
        
        parent_layout.addWidget(group_box)
    
    
    def create_extraction_group(self, parent_layout):
        """Create the metrics extraction group."""
        group_box = QGroupBox("Metrics Extraction")
        layout = QVBoxLayout(group_box)
        
        # Output path
        output_layout = QHBoxLayout()
        output_label = QLabel("Output CSV:")
        self.output_path = QLineEdit()
        self.output_path.setReadOnly(True)
        output_browse = QPushButton("Browse...")
        output_browse.clicked.connect(self.browse_output)
        
        output_layout.addWidget(output_label)
        output_layout.addWidget(self.output_path)
        output_layout.addWidget(output_browse)
        
        # Extract button
        button_layout = QHBoxLayout()
        self.extract_button = QPushButton("Extract Metrics")
        self.extract_button.clicked.connect(self.extract_metrics)
        self.extract_button.setEnabled(False)
        
        button_layout.addStretch(1)
        button_layout.addWidget(self.extract_button)
        
        layout.addLayout(output_layout)
        layout.addLayout(button_layout)
        
        parent_layout.addWidget(group_box)
    
    def create_status_section(self, parent_layout):
        """Create the status section."""
        layout = QHBoxLayout()
        
        self.status_label = QLabel("Ready")
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        layout.addWidget(self.status_label, 1)
        layout.addWidget(self.progress_bar, 2)
        
        parent_layout.addLayout(layout)
    
    def browse_project(self):
        """Open a dialog to select a project directory."""
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Project Directory", "", QFileDialog.Option.ShowDirsOnly
        )
        
        if dir_path:
            self.set_project_path(dir_path)
    
    def set_project_path(self, path):
        """Set the project path and update UI elements."""
        self.project_path = path
        self.path_input.setText(path)
        
        # Suggest an output path
        project_name = os.path.basename(path)
        suggested_output = os.path.join(
            os.path.dirname(path), 
            f"{project_name}_metrics.csv"
        )
        self.output_path.setText(suggested_output)
        
        # Enable the extract button if we have a project
        self.extract_button.setEnabled(True)
        
        # Update project info
        self.update_project_info()
        
        # Enable the structure group and populate if it's checked
        self.structure_group.setEnabled(True)
        if self.structure_group.isChecked():
            self.populate_project_tree(path)
        
        # Notify the parent window
        if self.parent and hasattr(self.parent, 'status_bar'):
            self.parent.status_bar.showMessage(f"Project loaded: {path}")
    
    def browse_output(self):
        """Open a dialog to select an output CSV file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Metrics CSV", "", "CSV Files (*.csv)"
        )
        
        if file_path:
            if not file_path.endswith('.csv'):
                file_path += '.csv'
            self.output_path.setText(file_path)
            
    
    def create_project_info_group(self, parent_layout):
        """Create the project information group."""
        group_box = QGroupBox("Project Information")
        layout = QVBoxLayout(group_box)
        
        # Project info display
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        
        # Add project structure tree with checkboxes in a collapsible section
        self.structure_group = QGroupBox("Project Structure Selection")
        self.structure_group.setCheckable(True)
        self.structure_group.setChecked(False)
        self.structure_group.toggled.connect(self.toggle_structure_section)
        
        structure_layout = QVBoxLayout(self.structure_group)
        
        # Create tree widget
        self.tree_widget = QTreeWidget()
        self.tree_widget.setHeaderLabels(["Directory"])
        self.tree_widget.header().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.tree_widget.setColumnCount(1)
        self.tree_widget.setAlternatingRowColors(True)
        self.tree_widget.setMaximumHeight(200)
        
        # Add buttons for easy selection
        selection_layout = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        deselect_all_btn = QPushButton("Deselect All")
        refresh_btn = QPushButton("Refresh Tree")
        
        select_all_btn.clicked.connect(self.select_all_dirs)
        deselect_all_btn.clicked.connect(self.deselect_all_dirs)
        refresh_btn.clicked.connect(lambda: self.populate_project_tree(self.project_path))
        
        selection_layout.addWidget(select_all_btn)
        selection_layout.addWidget(deselect_all_btn)
        selection_layout.addWidget(refresh_btn)
        
        structure_layout.addLayout(selection_layout)
        structure_layout.addWidget(self.tree_widget)
        
        # Add "Update Statistics" button
        update_stats_btn = QPushButton("Update Statistics")
        update_stats_btn.clicked.connect(self.update_statistics_with_selection)
        structure_layout.addWidget(update_stats_btn)
        
        # Add all widgets to the main layout
        layout.addWidget(self.info_text)
        layout.addWidget(self.structure_group)
        
        # Make the structure group visible but initially unchecked
        self.structure_group.setVisible(True)
        self.structure_group.setChecked(False)
        
        parent_layout.addWidget(group_box)
    
    def update_statistics_with_selection(self):
        """Update the file statistics based on selected directories."""
        if not self.project_path:
            return
        
        info_text = f"<h3>Project: {os.path.basename(self.project_path)}</h3>\n"
        info_text += f"<p><b>Location:</b> {self.project_path}</p>\n"
        
        # Count files by language
        c_cpp_count = 0
        java_count = 0
        python_count = 0
        other_count = 0
        total_files = 0
        
        # Get selected directories or use project path
        if self.structure_group.isChecked():
            selected_dirs = self.get_selected_directories()
            if not selected_dirs:  # If no directories are selected
                info_text += "<p><b>No directories selected.</b> Please select at least one directory.</p>\n"
                self.info_text.setHtml(info_text)
                return
            
            dirs_to_scan = selected_dirs
            info_text += "<p><b>Selected Directories:</b></p>\n<ul>\n"
            for dir_path in selected_dirs:
                info_text += f"<li>{os.path.relpath(dir_path, self.project_path)}</li>\n"
            info_text += "</ul>\n"
        else:
            # Default to entire project if structure selection not used
            dirs_to_scan = [self.project_path]
            # Common directories to exclude
            excluded_dirs = ['.git', '.svn', 'venv', '.env', '__pycache__', 
                            'node_modules', 'build', 'dist', '.idea', '.vscode']
        
        # Scan selected directories
        for dir_path in dirs_to_scan:
            for root, dirs, files in os.walk(dir_path):
                # Skip excluded directories if not using structure selection
                if not self.structure_group.isChecked():
                    dirs[:] = [d for d in dirs if d not in excluded_dirs]
                    
                for file in files:
                    total_files += 1
                    if file.endswith(('.c', '.cpp', '.h', '.hpp')):
                        c_cpp_count += 1
                    elif file.endswith('.java'):
                        java_count += 1
                    elif file.endswith('.py'):
                        python_count += 1
                    else:
                        other_count += 1
        
        info_text += "<h4>File Statistics:</h4>\n"
        info_text += f"<p>Total files: {total_files}</p>\n"
        info_text += f"<p>C/C++ files: {c_cpp_count}</p>\n"
        info_text += f"<p>Java files: {java_count}</p>\n"
        info_text += f"<p>Python files: {python_count}</p>\n"
        info_text += f"<p>Other files: {other_count}</p>\n"
        
        # Auto-select language based on file counts
        if c_cpp_count >= max(java_count, python_count):
            self.language_combo.setCurrentText("C/C++")
        elif java_count >= max(c_cpp_count, python_count):
            self.language_combo.setCurrentText("Java")
        elif python_count >= max(c_cpp_count, java_count):
            self.language_combo.setCurrentText("Python")
        
        self.info_text.setHtml(info_text)
        
        # Update status
        self.status_label.setText("Statistics updated based on selected directories")
        
    
    def toggle_structure_section(self, checked):
        """Toggle visibility of the tree and buttons within the structure section."""
        if checked and self.project_path:
            self.populate_project_tree(self.project_path)
            self.tree_widget.setVisible(True)
        else:
            self.tree_widget.setVisible(False)

    def populate_project_tree(self, root_path):
        """Populate the tree with project structure."""
        self.tree_widget.clear()
        
        # Excluded common directories that are typically not part of source code
        self.excluded_dirs = ['.git', '.svn', 'venv', '.env', '__pycache__', 
                            'node_modules', 'build', 'dist', '.idea', '.vscode']
        
        if not root_path or not os.path.isdir(root_path):
            return
        
        # Create root item
        root_item = QTreeWidgetItem(self.tree_widget, [os.path.basename(root_path)])
        root_item.setCheckState(0, Qt.CheckState.Checked)
        root_item.setExpanded(True)
        root_item.setData(0, Qt.ItemDataRole.UserRole, root_path)
        
        # Add directories to the tree
        self._add_directory(root_path, root_item)
        
        # Show the tree
        self.tree_widget.expandItem(root_item)

    def _add_directory(self, path, parent_item):
        """Recursively add directories to the tree."""
        try:
            for item in sorted(os.listdir(path)):
                full_path = os.path.join(path, item)
                if os.path.isdir(full_path):
                    # Skip common excluded directories by default
                    if item in self.excluded_dirs:
                        continue
                        
                    dir_item = QTreeWidgetItem(parent_item, [item])
                    dir_item.setCheckState(0, Qt.CheckState.Checked)
                    dir_item.setData(0, Qt.ItemDataRole.UserRole, full_path)
                    self._add_directory(full_path, dir_item)
        except (PermissionError, OSError) as e:
            self.logger.warning(f"Could not access directory {path}: {str(e)}")

    def select_all_dirs(self):
        """Select all directories in the tree."""
        self._set_check_state_recursive(self.tree_widget.invisibleRootItem(), Qt.CheckState.Checked)

    def deselect_all_dirs(self):
        """Deselect all directories in the tree."""
        self._set_check_state_recursive(self.tree_widget.invisibleRootItem(), Qt.CheckState.Unchecked)

    def _set_check_state_recursive(self, item, state):
        """Recursively set the check state of all items."""
        for i in range(item.childCount()):
            child = item.child(i)
            child.setCheckState(0, state)
            self._set_check_state_recursive(child, state)

    def get_selected_directories(self):
        """Get a list of selected directory paths."""
        selected_dirs = []
        self._get_checked_dirs_recursive(self.tree_widget.invisibleRootItem(), selected_dirs)
        return selected_dirs

    def _get_checked_dirs_recursive(self, item, selected_dirs):
        """Recursively collect checked directories."""
        for i in range(item.childCount()):
            child = item.child(i)
            if child.checkState(0) == Qt.CheckState.Checked:
                path = child.data(0, Qt.ItemDataRole.UserRole)
                if path:
                    selected_dirs.append(path)
            self._get_checked_dirs_recursive(child, selected_dirs)
        
    def update_project_info(self):
        """Update the project information display."""
        if self.structure_group.isChecked():
            # Use selected directories for statistics if structure selection is active
            self.update_statistics_with_selection()
        else:
            # Default behavior - scan the whole project with exclusions
            info_text = f"<h3>Project: {os.path.basename(self.project_path)}</h3>\n"
            info_text += f"<p><b>Location:</b> {self.project_path}</p>\n"
            
            # Count files by language with common exclusions
            excluded_dirs = ['.git', '.svn', 'venv', '.env', '__pycache__', 
                            'node_modules', 'build', 'dist', '.idea', '.vscode']
            
            c_cpp_count = 0
            java_count = 0
            python_count = 0
            other_count = 0
            total_files = 0
            
            for root, dirs, files in os.walk(self.project_path):
                # Skip excluded directories
                dirs[:] = [d for d in dirs if d not in excluded_dirs]
                
                for file in files:
                    total_files += 1
                    if file.endswith(('.c', '.cpp', '.h', '.hpp')):
                        c_cpp_count += 1
                    elif file.endswith('.java'):
                        java_count += 1
                    elif file.endswith('.py'):
                        python_count += 1
                    else:
                        other_count += 1
            
            info_text += "<h4>File Statistics:</h4>\n"
            info_text += f"<p>Total files: {total_files}</p>\n"
            info_text += f"<p>C/C++ files: {c_cpp_count}</p>\n"
            info_text += f"<p>Java files: {java_count}</p>\n"
            info_text += f"<p>Python files: {python_count}</p>\n"
            info_text += f"<p>Other files: {other_count}</p>\n"
            
            # Auto-select language based on file counts
            if c_cpp_count >= max(java_count, python_count):
                self.language_combo.setCurrentText("C/C++")
            elif java_count >= max(c_cpp_count, python_count):
                self.language_combo.setCurrentText("Java")
            elif python_count >= max(c_cpp_count, java_count):
                self.language_combo.setCurrentText("Python")
            
            self.info_text.setHtml(info_text)
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
        
        # Check if output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
            except OSError as e:
                QMessageBox.critical(self, "Error", f"Could not create output directory: {str(e)}")
                return
        
        # Get the selected language
        language = self.language_combo.currentText()
        
        # Disable UI elements during extraction
        self.extract_button.setEnabled(False)
        self.status_label.setText("Extracting metrics...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        
        # Create worker thread
        self.worker = ExtractionWorker(language, self.project_path, output_path)
        self.worker.finished.connect(self.on_extraction_finished)
        self.worker.error.connect(self.on_extraction_error)
        self.worker.start()
    
    def on_extraction_finished(self, csv_path):
        """Handle completion of metrics extraction."""
        self.metrics_path = csv_path
        self.progress_bar.setVisible(False)
        self.extract_button.setEnabled(True)
        self.status_label.setText(f"Metrics extracted to {csv_path}")
        
        # Show success message
        QMessageBox.information(
            self, 
            "Extraction Complete", 
            f"Metrics have been successfully extracted and saved to:\n{csv_path}"
        )
        
        # Update parent window if it exists
        if self.parent and hasattr(self.parent, 'status_bar'):
            self.parent.status_bar.showMessage(f"Metrics extracted: {csv_path}")
            
            # Switch to metrics tab if it exists
            if hasattr(self.parent, 'tabs') and hasattr(self.parent, 'metrics_tab'):
                self.parent.metrics_tab.load_metrics(csv_path)
                self.parent.tabs.setCurrentWidget(self.parent.metrics_tab)
    
    def on_extraction_error(self, error_msg):
        """Handle errors during metrics extraction."""
        self.progress_bar.setVisible(False)
        self.extract_button.setEnabled(True)
        self.status_label.setText("Extraction failed")
        
        QMessageBox.critical(self, "Extraction Error", error_msg)
        
        if self.parent and hasattr(self.parent, 'status_bar'):
            self.parent.status_bar.showMessage("Metrics extraction failed")