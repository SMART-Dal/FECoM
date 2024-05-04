import os

terms = [
    ["instrumentation", "overhead", "performance", "code", "energy"],
    ["measurement", "noise", "process", "profiling"],
    ["hardware", "configuration", "device", "power"],
    ["calibration"],
    ["GPU", "optimization", "data"],
    ["RAPL", "precision"],
    ["frequency"],
    ["patch", "correctness", "API"],
    ["coverage", "identification"],
    ["compatibility", "version", "dependency"],
    ["memory", "CUDA", "allocation", "management"],
    ["Docker", "container"]
]

base_dir = "query_results"

for term_list in terms:
    for term in term_list:
        folder_path = os.path.join(base_dir, term)
        try:
            os.makedirs(folder_path)
            print(f"Created folder: {folder_path}")
        except FileExistsError:
            print(f"Folder already exists: {folder_path}")