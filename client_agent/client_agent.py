import os
import subprocess

input_dir = './code-dataset/input-code'
output_dir = './code-dataset/patched-code/'
script_path = 'script_patcher.py'

# delete all the files in the output directory before running the script
for filename in os.listdir(output_dir):
    file_path = os.path.join(output_dir, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
    except Exception as e:
        print('Failed to delete %s ,for the Reason: %s' % (file_path, e))

# run the script for all the files in the input directory and save the patched files in the output directory
for file in os.listdir(input_dir):
    if file.endswith('.py'):
        input_file_path = os.path.join(input_dir, file)
        print("File path is:",input_file_path)

        # create the patched file
        input_name = os.path.splitext(input_file_path.split('/')[-1])[0] # get the script name
        output_path = output_dir + input_name + '-Patched.py'
        with open(output_path, 'w') as f:
            subprocess.run(['python3', script_path, input_file_path], stdout=f)

