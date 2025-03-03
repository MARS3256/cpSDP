#!/usr/bin/env python3
# main.py - Entry point for the Software Defect Prediction Tool

import sys
import os
import logging
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QIcon
from ui.main_window import MainWindow
from core.utils import load_config

def setup_logging():
    """Configure logging for the application."""
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, 'cpSDP.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting cpSDP - Software Defect Prediction Tool")
    
    return logger

def main():
    """Main entry point for the application."""
    # Set up logging
    logger = setup_logging()
    
    # Load configuration
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config', 'config.json')
    if os.path.exists(config_path):
        config = load_config(config_path)
    else:
        logger.warning(f"Configuration file not found: {config_path}")
        config = {}
    
    # Create the application
    app = QApplication(sys.argv)
    app.setApplicationName("cpSDP")
    app.setOrganizationName("cpSDP")
    
    # Set application style
    app.setStyle("Fusion")
    
    # Create and show the main window
    main_window = MainWindow()
    main_window.show()
    
    # Start the event loop
    sys.exit(app.exec())

if __name__ == "__main__":
    main()