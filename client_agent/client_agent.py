import os
import subprocess
import shutil

input_dir = './code-dataset/input-code'
output_dir = './code-dataset/patched-code/'
script_path = 'script_patcher.py'

for file in os.listdir(input_dir):
    if file.endswith('.py'):
        input_file_path = os.path.join(input_dir, file)
        print("File path is:",input_file_path)
        input_name = os.path.splitext(input_file_path.split('/')[-1])[0] # get the script name
        output_path = output_dir + input_name + '-Patched.py'
        with open(output_path, 'w') as f:
            subprocess.run(['python3', script_path, input_file_path], stdout=f)