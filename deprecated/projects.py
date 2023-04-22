# define all projects to run

import os
import random
from pathlib import Path
from tool.client.client_config import EXPERIMENT_DIR
from tool.experiment.experiments import ExperimentKinds


def get_all_projects():
    project_dir = EXPERIMENT_DIR/ExperimentKinds.METHOD_LEVEL.value
    parent_projects = sorted(os.listdir(project_dir))
    # list of lists
    all_projects = [[f"{parent}/{child}" for child in sorted(os.listdir(project_dir/parent))] for parent in parent_projects]
    return all_projects

def pick_random_projects(available_projects):
    random_projects = []
    for child_projects in available_projects:
        # pop a random child project out of the list
        random_child = child_projects.pop(random.randrange(len(child_projects)))
        random_projects.append(random_child)
    return random_projects

def write_projects_to_file(file_path: Path, projects: list):
    with open(file_path, 'w') as f:
        f.writelines([f"{project}\n" for project in projects])

if __name__ == "__main__":
    random.seed(42)
    all_available_projects = get_all_projects()
    random_projects = pick_random_projects(all_available_projects)
    write_projects_to_file("out/initial_projects.txt", random_projects)