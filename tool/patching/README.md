# Client Agent Program for Code Energy Consumption

This is the client agent that takes a python program along with list of libraries/function they want to calculate energy consumption for, as inputs. The client agent then runs its analyzer script to extract the function calls using an Abstract Syntax Tree(AST) and creates Request Packets for these calls. Then it filters out the functions that the client/user wants the energy consumption to be calculated for. It will send these filtered request packets to the Server which will run the method,calculate the energy consumption and then send back the required energy data packet for the method calls, as a response packet.

## Usage

<!-- Read a bit about measuring Software Energy Consumption and how it is done in Linux using the Profiling Tool known as [Perf](https://perf.wiki.kernel.org/index.php/Main_Page). -->

The `~/data/code-dataset` directory (`~/data` indicates that `data` is in the top-level directory of this repository) contains the dataset for which we want to calculate the energy consumption for. Make sure it has two subdirectories:
- `Repositories`: this contains the client projects
- `Patched-Repositories`: this will contain the patched version of the same client projects after running the patching script.  
If needed, create these two subdirectories using `mkdir`.
  
During the patching process we parse the python code into an AST using the 'ast' library, make changes in the scripts to identify the target method calls and create request packets for the same. Once the patching process is complete, we will have the final scripts ready, which on running will produce energy consumption data for the identified libraries for the client.
Step 1 is to move or clone your target project into the `~/data/code-dataset/Repositories` directory.
```bash
git clone git_repo_link
```
Once we have the target repo, we can run the patching script. 
```bash
python3 repo-patching.py
```

The final patched code will be stored in the `~/data/code-dataset/Patched-Repositories` directory. You can modify the paths for these directories in "patching_config.py".

Once the repo is patched, it is important that you update the `EXPERIMENT_DIR` and `EXPERIMENT_TAG` settings in `patching_config.py`. Then you can simply run the python files from the patched projects like any a script with `python3 project_name.py`. When executing, the patched files will send method requests to the server, get the energy consumption data and store it in `~/data/energy-dataset`.

Once the execution is complete we will have a list of json objects for each method call. Please refer to `~/tool/measurement/README.md` for more information on the format of these objects.


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Thank you!!!
