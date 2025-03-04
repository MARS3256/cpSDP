# cpSDP - Software Defect Prediction Tool

## Overview
cpSDP is a research project focused on the development and analysis of Software Defect Prediction (SDP) techniques. This cross-platform tool analyzes code metrics to predict potential defects in C/C++, Java, and Python projects, helping developers identify areas that may need attention before they become problematic.

## Features
- Cross-language support for C/C++, Java, and Python projects
- Extraction of 18 code quality metrics including:
  - CBO (Coupling Between Objects)
  - DIT (Depth Inheritance Tree)
  - RFC (Response For a Class)
  - LOC (Lines of Code)
  - and many more
- Interactive data visualization of metrics
- Machine learning-based defect prediction
- Detailed reporting and analysis of defect-prone areas
- User-friendly PyQt6 interface

## Installation

### Prerequisites
- Python 3.8 or higher
- Git (for cloning the repository)

### Step 1: Clone the Repository
```sh
git clone https://github.com/yourusername/cpSDP.git
cd cpSDP
```

### Step 2: Create a Virtual Environment (Optional but Recommended)
```sh
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```sh
pip install -r requirements.txt
```

## Usage

### Running the Application
```sh
python main.py
```

### Workflow
1. Select a project directory to analyze
2. Extract metrics for the selected project
3. Train a defect prediction model using the metrics
4. Apply the model to predict defects in other projects
5. View detailed reports and visualizations

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Author
Suravi Akhter
Faculty Member
ULAB, CSE Department

Muhammad As Adur Rahman Sajid  
ULAB, CSE Department  
Student ID: 233014037  
[Github](https://github.com/MARS3256/)




All rights reserved

## Contributing
Contributions to improve cpSDP are welcome. Please feel free to submit pull requests or open issues to enhance the functionality.

## Contact
For any inquiries, please contact [mars.vxi3@gmail.com](mailto:mars.vxi3@gmail.com).
