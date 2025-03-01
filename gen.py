import os

def create_folder_tree(folder_map, root_dir='.'):
    """
    Create a folder tree based on a dictionary of strings.

    Args:
        folder_map (dict): A dictionary where keys are folder names and values are lists of subfolders or files.
        root_dir (str): The root directory where the folder tree will be created. Defaults to current directory.

    Returns:
        None
    """
    for folder, subfolders in folder_map.items():
        folder_path = os.path.join(root_dir, folder)
        if '.' in folder:  # Check if it's a file
            with open(folder_path, 'w') as f:
                pass  # Create an empty file
        else:
            os.makedirs(folder_path, exist_ok=True)
            if isinstance(subfolders, dict):
                create_folder_tree(subfolders, folder_path)
            elif isinstance(subfolders, list):
                create_folder_tree({subfolder: [] for subfolder in subfolders}, folder_path)

# Example usage:
# folder_map = {
#     'Actions': ['action.py'],
#     'AIAgents': {
#         'AIData': ['memoryData.py', 'synData.py', 'contextRetriever.py', 'memorySaver.py', 'summarizer.py', 'synAI.py']
#     },
#     'ControlPanel': ['hub.py', 'promptInput.py'],
#     'Data': ['userMemories.json'],
#     'scratchpad': ['questions.txt', 'test.txt'],
#     'tests': ['chat.py', 'newchat.py', 'ntts.py', 'test.py'],
#     'Widgets': ['cardHolder.py', 'cards.py', 'VScrollView.py'],
#     '.env': [],
#     'synVoice.py': [],
#     't.py': [],
#     'twitchChat.py': [],
#     'v.py': []
# }

folder_map = {
    'main.py': [],
    'requirements.txt': [],
    'README.md': [],
    'config': {
        'config.json': []
    },
    'core': {
        '__init__.py': [],
        'feature_extractor.py': [],
        'model_trainer.py': [],
        'predictor.py': [],
        'utils.py': []
    },
    'extractors': {
        '__init__.py': [],
        'c_cpp_extractor.py': [],
        'java_extractor.py': [],
        'python_extractor.py': []
    },
    'ui': {
        '__init__.py': [],
        'main_window.py': [],
        'project_tab.py': [],
        'metrics_tab.py': [],
        'prediction_tab.py': []
    },
    'tests': {
        '__init__.py': [],
        'test_extractors.py': [],
        'test_model.py': [],
        'test_ui.py': []
    }
}

create_folder_tree(folder_map)