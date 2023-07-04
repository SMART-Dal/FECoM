import os
import shutil
import json

# source_dir = '/home/saurabh/code-energy-consumption/data/code-dataset/Patched-Repositories/tutorials'
# dest_dir = '/home/saurabh/code-energy-consumption/data/energy-dataset'

# # Recursively find all Jupyter notebooks in the source directory
# notebooks = []
# for root, dirs, files in os.walk(source_dir):
#     for file in files:
#         if file.endswith('.ipynb'):
#             notebooks.append(os.path.join(root, file))

# # Create a new subdirectory for each notebook in the destination directory
# for notebook in notebooks:
#     notebook_name = os.path.splitext(os.path.basename(notebook))[0]
#     dest_subdir = os.path.join(dest_dir, os.path.basename(os.path.dirname(notebook)), notebook_name)
#     os.makedirs(dest_subdir, exist_ok=True)
    
#     # Create 10 JSON files in the new subdirectory
#     for i in range(1, 11):
#         experiment_file_path = os.path.join(dest_subdir, f'experiment-{i}.json')
#         with open(experiment_file_path, 'w') as f:
#             json.dump([], f)


for i in range(1, 11):
    experiment_dir = "/users/grad/srajput/GreenAI-extension/data/energy-dataset/data-size/images/cnn/"
    os.makedirs(experiment_dir, exist_ok=True)
    experiment_file_path = os.path.join(experiment_dir, f'experiment-{i}.json')
    with open(experiment_file_path, 'w') as f:
        json.dump([], f)
