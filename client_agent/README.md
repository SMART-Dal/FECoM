# Client Agent Program for Code Energy Consumption

This is the client agent that takes a python program along with list of libraries/function they want to calculate energy consumption for, as inputs. The client agent then runs its analyzer script to extract the function calls using an Abstract Syntax Tree(AST) and creates Request Packets for these calls. Then it filters out the functions that the client/user wants the energy consumption to be calculated for. It will send these filtered request packets to the Server which will run the method,calculate the energy consumption and then send back the required energy data packet for the method calls, as a response packet.

## Usage

<!-- Read a bit about measuring Software Energy Consumption and how it is done in Linux using the Profiling Tool known as [Perf](https://perf.wiki.kernel.org/index.php/Main_Page). -->

The "code-dataset" directory contains the dataset for which we want to calculate the energy consumption for. It has two subdirectories: "Repositories" which contains the client projects and "Patched-Repositories" that contains patched version of the same client projects. During the patching process we parse the python code into an AST using 'ast' library, make changes in the scripts to identify the target method calls and create request packets for the same. Once the patching process is complete, we will have the final scripts ready, which on running will produce energy consumption data for the identified libraries for the client.
Step 1 is to move or clone your target project into the "Repositories" directory.
```bash
git clone git_repo_link
```
Once we have the target repo, we can run the patching script. 
```bash
python repo-patching.py
```

The final patched code will be stored in the "Patched-Repositories" directory. You can modify the paths for these directories from "clientconfig.py".

Once we have the patched repo ready, we can simply run python files from these projects as a normal script. On execution these files will send method requests to the server, get the energy consumption data and store it in "methodcall-energy-dataset.json" file.

Once the execution is complete we will have a list of json objects for each method call. Please refer to `~/server/README.md` for more information on the format of these objects.


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Thank you!!!
