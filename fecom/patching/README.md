# Patching Module

The Patching module, included in the "fecom" directory of the replication package, serves as the foundation of our framework, providing two types of patching scripts: one for the method level and another for the project level. Both are integrated into the "repo-patching" program. This module enables users to provide Python scripts and specify target frameworks as inputs for energy measurement, allowing precise energy measurement at both the method and project levels.

## Usage

1. Ensure you have the necessary dataset located in the `~/data/code-dataset` directory. This directory contains two subdirectories:
   - "Repositories": Contains the client projects (unpatched versions).
   - "Patched-Repositories": Will contain the patched versions of the client projects after running the patching script.

   If the subdirectories do not exist, create them using `mkdir`.

2. Move or clone your target project for which you want to calculate energy consumption into the `~/data/code-dataset/Repositories` directory:
   ```bash
   git clone git_repo_link
   ```

3.  Add the target library or libraries you want to calculate energy consumption for in the `method_level_script_patcher.py` file by adding them to the `requiredLibraries` list, for example: `requiredLibraries = ["tensorflow"]`.

4. Run the patching script `repo-patching.py`:
   ```bash
   python3 repo-patching.py
   ```

   The patching script parses the Python code into an Abstract Syntax Tree (AST) using the 'ast' library. It identifies the target method calls, creates request packets, and generates the final patched scripts stored in the `~/data/code-dataset/Patched-Repositories` directory.

5. Update the `EXPERIMENT_DIR` and other relevant paths specific to your machine settings in `patching_config.py`.

6. Execute the Python files from the patched projects as you would normally run a script:
   ```bash
   python3 project_name.py
   ```

   The patched files will send method requests to the measurement module, retrieve the energy consumption data, and store it in `~/data/energy-dataset`.

7. After execution, you will have a list of JSON objects for each method call. For more information on the format of these objects, refer to `~/fecom/measurement/README.md`.

Thank you!
