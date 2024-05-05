import os

# terms = [
#     ["instrumentation", "overhead", "performance", "code", "energy"],
#     ["measurement", "noise", "process", "profiling"],
#     ["hardware", "configuration", "device", "power"],
#     ["calibration"],
#     ["GPU", "optimization", "data"],
#     ["RAPL", "precision"],
#     ["frequency"],
#     ["patch", "correctness", "API"],
#     ["coverage", "identification"],
#     ["compatibility", "version", "dependency"],
#     ["memory", "CUDA", "allocation", "management"],
#     ["Docker", "container"]
# ]

# base_dir = "query_results"

# for term_list in terms:
#     for term in term_list:
#         folder_path = os.path.join(base_dir, term)
#         try:
#             os.makedirs(folder_path)
#             print(f"Created folder: {folder_path}")
#         except FileExistsError:
#             print(f"Folder already exists: {folder_path}")

# ---------------------------------------------

# from rq3_data import challenges
# import json

# def create_json_files(challenges):
#     for category in challenges:
#         for subcategory in category["subcategories"]:
#             for challenge in subcategory["challenges"]:
#                 folder_name = f"c{challenge['query_id']}"
#                 folder_path = os.path.join("replication/rq3_analysis/filtered_posts", folder_name)
#                 # os.makedirs(folder_path, exist_ok=True)

#                 file_path = os.path.join(folder_path, "mitigation_strategies.json")

#                 mitigation_strategies = []

#                 # create JSON file
#                 with open(file_path, "w") as file:
#                     json.dump(mitigation_strategies, file, indent=2)

# print("JSON files created successfully.")

# create_json_files(challenges)

# ---------------------------------------------

from rq3_data import challenges
import json
import os
import pandas as pd

def create_excel_file(challenges):
    data = []

    for category in challenges:
        for subcategory in category["subcategories"]:
            for challenge in subcategory["challenges"]:
                folder_name = f"c{challenge['query_id']}"
                folder_path = os.path.join("./filtered_posts/", folder_name)
                file_path = os.path.join(folder_path, "mitigation_strategies.json")
                print(file_path)
                if os.path.exists(file_path):
                    with open(file_path, "r") as file:
                        mitigation_strategies = json.load(file)

                    for strategy in mitigation_strategies:
                        data.append({
                            "Category": category["category"],
                            "Query ID": challenge["query_id"],
                            "Subcategory": subcategory["name"],
                            "Challenge": challenge["name"],
                            "Mitigation Strategy": strategy["mitigation"],
                            "Document IDs": ", ".join(strategy["document_ids"])
                        })

    df = pd.DataFrame(data)
    excel_file_path = "challenges_mitigation_analysis.xlsx"
    df.to_excel(excel_file_path, index=False)

    print(f"Excel file '{excel_file_path}' created successfully.")

create_excel_file(challenges)

import json
import os

# Directory containing the subfolders
directory = "./filtered_posts/"

# Iterate over each subfolder
for subfolder in ["c" + str(i) for i in range(1, 13)]:
    subfolder_path = os.path.join(directory, subfolder)
    
    # Check if the subfolder exists
    if os.path.isdir(subfolder_path):
        # Mitigation strategies file
        mitigation_strategies_file = os.path.join(subfolder_path, "mitigation_strategies.json")
        
        # Check if the mitigation_strategies.json file exists
        if os.path.isfile(mitigation_strategies_file):
            # Read the mitigation_strategies.json file
            with open(mitigation_strategies_file, "r") as file:
                mitigation_strategies = json.load(file)
            
            # Corresponding ranked_results.json file
            ranked_results_file = os.path.join(subfolder_path, "ranked_results.json")
            
            # Check if the ranked_results.json file exists
            if os.path.isfile(ranked_results_file):
                # Read the ranked_results.json file
                with open(ranked_results_file, "r") as file:
                    ranked_results = json.load(file)
                
                # Extract all document_ids from ranked_results
                ranked_document_ids = [result["document_id"] for result in ranked_results if "document_id" in result]
                
                # Check if each document_id in mitigation_strategies exists in ranked_results
                for mitigation in mitigation_strategies:
                    for document_id in mitigation["document_ids"]:
                        if document_id not in ranked_document_ids:
                            print(f"Subfolder: {subfolder}")
                            print(f"Mitigation: {mitigation['mitigation']}")
                            print(f"Document ID: {document_id}")
                            print("Not found in ranked_results.json")
                            print()
            else:
                print(f"File not found: {ranked_results_file}")
        else:
            print(f"File not found: {mitigation_strategies_file}")
    else:
        print(f"Subfolder not found: {subfolder_path}")