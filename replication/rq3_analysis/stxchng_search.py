import requests
import json
import time
import signal
import sys

# Set the base URL for the Stack Exchange API
base_url = 'https://api.stackexchange.com/2.3'

# Set your access token
access_token = ''
key = ''

# Global variables to store the current results, term, and site
current_results = []
current_term = None
current_site = None

# Function to handle keyboard interrupt
def signal_handler(sig, frame):
    print("\nKeyboard interrupt detected. Saving results so far...")
    save_results(current_results, current_term, current_site)
    sys.exit(0)

# Function to save results to a file
def save_results(results, term, site):
    with open(f'query_results/{term}/{site}_results.json', 'w', encoding='utf-8') as file:
        json.dump(results, file, ensure_ascii=False, indent=4)
    print(f"Results have been written to 'query_results/{term}/{site}_results.json'")

# Function to fetch and process search results
def fetch_and_process_results(search_params):
    print(f"Fetching results for page {search_params['page']}...")
    while True:
        search_response = requests.get(f'{base_url}/search/advanced', params=search_params)
        search_data = search_response.json()
        # print(search_data)

        if 'backoff' in search_data:
            backoff_time = search_data['backoff']
            print(f"Backoff encountered. Waiting for {backoff_time} seconds before retrying...")
            time.sleep(backoff_time+10)
        elif 'error_id' in search_data and search_data['error_id'] == 502:
            print("Throttle violation encountered. Waiting for 60 seconds before retrying...")
            time.sleep(60)
        else:
            break

    result = []

    # Process the search results
    for question in search_data['items']:
        result.append(question)

    return result, search_data['has_more']

# Set up signal handler for keyboard interrupt
signal.signal(signal.SIGINT, signal_handler)

# term = 'performance'
lf = [[
    #'code', 
    # 'configuration', 'device', 
       #'optimization', 
       #'data', 'API', 'version', 'coverage', 'identification',
       #'compatibility','dependency' , 'allocation', 'management', 
       #'correctness', 
    #    'container'
       ]]
lf_done = ["TensorFlow", 
       'Docker', 
       'memory']
site = 'stackoverflow'
terms = [
    # [ "code", "energy"],
    # ["measurement", "noise", "process", "profiling"],
    #[ 
    #    "hardware", 
    #    "configuration", 
    #    "device", 
    #    "power"],
    # ["calibration"],
    #[
     #   "GPU", 
     #   "optimization", 
     #   "data"
      #  ],
    #["RAPL", "precision"],
    #["frequency"],
    #["patch", 
    # "correctness", 
    # "API"
    # ],
    #["coverage", "identification"],
    #["compatibility", 
     #"version", 
    # "dependency"],
    #[
     #   "memory", 
     #"CUDA", 
     #"allocation", 
     #"management"
     #],
    [
    # "Docker", 
    # "container"
     ]
]

for term_list in lf:
    for term in term_list:
        # Set the search parameters
        search_params = {
            'q': term,
            'accepted': 'True',
            'site': site,
            # 'sort': 'votes',
            'order': 'desc',
            'pagesize': 100,
            'page': 1,
            'access_token': access_token,  # Include the access token
            'key': key,
            'filter': '!)sBhEMd_srY6nQkl(_R-'
        }
        print(f"Searching for term: {term}")
        current_results = []
        current_term = term
        current_site = site

        # Fetch and process the initial results
        results, has_more = fetch_and_process_results(search_params)
        current_results.extend(results)

        # Fetch and process additional results until has_more becomes False
        while has_more:
            search_params['page'] += 1
            additional_results, has_more = fetch_and_process_results(search_params)
            current_results.extend(additional_results)

        # Write the results to a JSON file with name as term variable + _results.json inside query_results folder
        # Save the results to a file
        save_results(current_results, current_term, current_site)

        print("Results have been written to 'results.json' for term: ", term)
        time.sleep(120)