import requests
import random
from ragatouille import RAGTrainer
from ragatouille.data import CorpusProcessor, llama_index_sentence_splitter
from ragatouille import RAGPretrainedModel
import os
import json
import faiss
import torch

challenges = [
  {
    "category": "Energy measurement",
    "subcategories": [
      {
        "name": "Instrumentation challenges",
        "challenges": [
          {
            "name": "Instrumentation overhead",
            "description": "Instrumented code has additional instructions that may account for overhead and therefore may impact the performance and energy consumption of the code.",
            "mitigation": [],
            "keywords": ["instrumentation", "overhead", "performance", "code", "energy"],
            "query_id": 1,
            "average_similarity": 0.4478
          },
          {
            "name": "Noise in measurement",
            "description": "Background processes running on the machine during energy measurement introduce noise and overheads, affecting the accuracy of measured energy.",
            "mitigation": [],
            "keywords": ["energy", "measurement", "noise", "process", "profiling"],
            "query_id": 2,
            "average_similarity": 0.3844
          }
        ]
      },
      {
        "name": "Hardware variability",
        "challenges": [
          {
            "name": "Hardware configuration",
            "description": "Different hardware configurations can lead to variations in energy consumption values for the same project.",
            "mitigation": [],
            "keywords": ["hardware", "energy", "configuration", "device", "power"],
            "query_id": 3,
            "average_similarity": 0.3357
          },
          {
            "name": "Calibration issues",
            "description": "Energy measurement tools require calibration to account for hardware variations.",
            "mitigation": [],
            "keywords": ["calibration", "energy", "measurement", "hardware", "power"],
            "query_id": 4,
            "average_similarity": 0.3616
          },
          {
            "name": "GPU usage",
            "description": "The selected subject systems must utilize GPU in an optimized manner. Failing to ensure this challenge may introduce incorrect and inconsistent energy data collection.",
            "mitigation": [],
            "keywords": ["GPU", "energy", "optimization", "data", "performance"],
            "query_id": 5,
            "average_similarity": 0.3195
          }
        ]
      },
      {
        "name": "Granularity of energy attribution",
        "challenges": [
          {
            "name": "Precision limits",
            "description": "Existing software tools, due to Intel RAPL limitation, do not permit energy measurements at intervals smaller than 1ms.",
            "mitigation": [],
            "keywords": ["energy", "RAPL", "measurement", "precision", "power"],
            "query_id": 6,
            "average_similarity": 0.3469
          },
          {
            "name": "Precision overhead balance",
            "description": "Observing energy consumption at a high frequency improves the precision of observed data; however, such high frequencies also introduce computation overhead that may introduce noise.",
            "mitigation": [],
            "keywords": ["energy", "precision", "overhead", "frequency", "noise"],
            "query_id": 7,
            "average_similarity": 0.3741
          }
        ]
      }
    ]
  },
  {
    "category": "Patching",
    "subcategories": [
      {
        "name": "Patch generation",
        "challenges": [
          {
            "name": "Correctness of patches",
            "description": "Each identified patch location (in our case, each TensorFlow API) must be correctly patched to record the correct energy consumption of the API and not introduce new syntactic or semantic issues.",
            "mitigation": [],
            "keywords": ["energy", "patch", "TensorFlow", "correctness", "API"],
            "query_id": 8,
            "average_similarity": 0.4110
          },
          {
            "name": "Patch coverage",
            "description": "Each patch location must be identified correctly to avoid missing code that is supposed to be patched and measured.",
            "mitigation": [],
            "keywords": ["patch", "code", "coverage", "identification", "energy"],
            "query_id": 9,
            "average_similarity": 0.3602
          }
        ]
      }
    ]
  },
  {
    "category": "Execution environment",
    "subcategories": [
      {
        "name": "Hardware incompatibility",
        "challenges": [
          {
            "name": "Hardware incompatibility",
            "description": "Compatibility issues arise when using framework versions (e.g. TensorFlow) that are not compatible with the machine's hardware or software dependencies.",
            "mitigation": [],
            "keywords": ["compatibility", "hardware", "version", "TensorFlow", "dependency"],
            "query_id": 10,
            "average_similarity": 0.3068
          }
        ]
      },
      {
        "name": "GPU challenges",
        "challenges": [
          {
            "name": "Memory management",
            "description": "CUDA memory allocation errors may arise when a process cannot allocate sufficient memory during data processing on a GPU.",
            "mitigation": [],
            "keywords": ["memory", "GPU", "CUDA", "allocation", "management"],
            "query_id": 11,
            "average_similarity": 0.4511
          },
          {
            "name": "Container issues",
            "description": "Incompatibility of Docker containers with specific GPUs and TensorFlow versions may hinder the replication of a project.",
            "mitigation": [],
            "keywords": ["Docker", "container", "GPU", "TensorFlow", "version"],
            "query_id": 12,
            "average_similarity": 0.3467
          }
        ]
      }
    ]
  }
]

def aggregate_stackoverflow_posts(path):
    aggregated_so_posts = []
    unique_question_ids = set()

   # add tqdm for progress bar
    for root, dirs, files in os.walk(path):
        for file in files:
            if file == "stackoverflow_results.json":
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r") as f:
                        posts = json.load(f)
                        for post in posts:
                            question_id = post["question_id"]
                            if question_id not in unique_question_ids:
                                unique_question_ids.add(question_id)
                                aggregated_so_posts.append(post)
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")
    with open("aggregated_so_posts.json", "w") as f:
        json.dump(aggregated_so_posts, f, indent=4)

    print("Aggregated Stack Overflow posts saved to 'aggregated_so_posts.json'")
    print(f"Number of unique questions: {len(unique_question_ids)}")

def get_stackoverflow_page():
    # read list from fo;e aggregated_so_posts.json which contains a list of objects, each object is a stackoverflow post
    with open("aggregated_so_posts.json", "r") as f:
        posts = json.load(f)
    return posts

def prepare_corpus():
    my_full_corpus = get_stackoverflow_page()
    documents = []
    try:
        for post in my_full_corpus:
            #print(post['answers'][0]["body_markdown"])
            documents.append(post['answers'][0]["body_markdown"])
    except Exception as e:
            print("Error processing post: ", e)
    # write the documents to a json file
    with open("corpus.json", "w") as f:
        json.dump(documents, f, indent=4)
    return documents

def prepare_queries():
    queries = []
    for post in get_stackoverflow_page():
        queries.append(post["title"] + " " + post["body_markdown"])
    return queries

def create_query_document_from_stackoverflow_object(stackoverflow_object):
    query = stackoverflow_object["title"] + " " + stackoverflow_object["body_markdown"]
    document = stackoverflow_object["answers"][0]["body_markdown"]
    # posts have accepted answers for each post asking question
    return query, document

def prepare_queries_and_documents():
    stackoverflow_posts = get_stackoverflow_page()
    query_document_pairs = []
    for post in stackoverflow_posts:
        query, document = create_query_document_from_stackoverflow_object(post)
        query_document_pairs.append((query, document))
    return query_document_pairs

def finetune_colbert():
    trainer = RAGTrainer(model_name="FeCOMColBERT", pretrained_model_name="colbert-ir/colbertv2.0", language_code="en")
    
    #documents = prepare_corpus()
    #queries = prepare_queries()
    pairs = prepare_queries_and_documents()
    
    trainer.prepare_training_data(raw_data=pairs, data_out_path="./data/", all_documents=documents, num_new_negatives=10, mine_hard_negatives=True)
    
    trainer.train(batch_size=32,
                  nbits=4,
                  maxsteps=500000,
                  use_ib_negatives=True,
                  dim=128,
                  learning_rate=5e-6,
                  doc_maxlen=256,
                  use_relu=False,
                  warmup_steps="auto")

def index_posts():
    # read from corpus.json which has a list of documents
    with open("corpus.json", "r") as f:
        corpus = json.load(f)

    print("Indexing documents...")  
    print("Number of documents: ", len(corpus))
    RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
    RAG.index(
    collection=corpus,
    index_name="FECOM", 
    split_documents=True
    )

    k = 3 # How many documents you want to retrieve, defaults to 10, we set it to 3 here for readability
    results = RAG.search(query="What are issues in energy consumption measurement?", k=k)
    print(results)

def search_index(query, k=50):
    #torch.cuda.empty_cache()

    # This is the path to index. We recommend keeping this path format when using RAGatouille somewhere else.
    path_to_index = ".ragatouille/colbert/indexes/FECOM/"
    RAG = RAGPretrainedModel.from_index(path_to_index)
    search_results = RAG.search(
        query=query,
        index_name="FECOM",
        k=k
    )
    return search_results

def get_queries():
    docs = []
    for category in challenges:
        for subcategory in category['subcategories']:
            for challenge in subcategory['challenges']:
                challenge_id = challenge['query_id']
                old_system_prompt = '''
                Measuring energy consumption at lower granularities, such as API-
                level poses unique challenges compared to coarse-grained measurement. This research question aims to support developers and researchers working in this field by
                elaborating on the issues, considerations, and challenges one may encounter while developing a tool to measure energy consumption. What are the considerations, solutions and mitigation strategies for the following challenges?: '''
                system_prompt = '''What are solutions when challanges like following occurs?:'''
                docs.append((challenge_id,f"{subcategory['name']}: {challenge['name']}: {challenge['description']}"))
    return docs

if __name__ == "__main__":
    # Usage
    #post_path = "/home/saurabh/code-energy-consumption/replication/rq3_analysis/query_results"
    #aggregate_stackoverflow_posts(path)
    #finetune_colbert()
    #prepare_corpus()
    #index_posts()
    search_query = get_queries()
    queries = []
    for query in search_query:
        queries.append(query[1])
    
    results = search_index(queries)
    for index, result in enumerate(results):
        print(result)
        #write tp file
        with open(f"c{index+1}/ranked_results.json", "w") as f:
            json.dump(result, f, indent=4)
        
    #print(json.dumps(results, indent=4))
