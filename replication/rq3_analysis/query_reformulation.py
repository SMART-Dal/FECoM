from keybert import KeyBERT
from rq3_data import challenges
import numpy as np
from typing import List, Tuple
import json
import signal

kw_model = KeyBERT(model='flax-sentence-embeddings/stackoverflow_mpnet-base')

def calculate_average_similarity(keywords):
    similarities = [keyword[1] for keyword in keywords]
    return np.mean(similarities)

docs = []
for category in challenges:
    for subcategory in category['subcategories']:
        for challenge in subcategory['challenges']:
            docs.append(f"{challenge['name']}: {challenge['description']}")

def get_threshold(docs):
    keywords_list = kw_model.extract_keywords(docs, keyphrase_ngram_range=(1, 1))

    index = 0
    for category in challenges:
        print(f"Category: {category['category']}")
        for subcategory in category['subcategories']:
            print(f"\tSubcategory: {subcategory['name']}")
            for challenge in subcategory['challenges']:
                keywords = keywords_list[index]
                avg_similarity = calculate_average_similarity(keywords)
                print(f"\t\tChallenge: {challenge['name']}")
                print(f"\t\tKeywords: {keywords}")
                print(f"\t\tAverage Cosine Similarity: {avg_similarity:.4f}")
                print()
                index += 1

def save_results(tag_files, challenge_id, signal_received=None):
    """
    Save the relevant posts to a JSON file.
    
    Args:
        relevant_posts (dict): Dictionary containing relevant posts for each challenge.
        signal_received (int, optional): Signal number received (default is None).
    """
    with open(f'filtered_posts/c{challenge_id}/tags_files.json', 'w', encoding='utf-8') as file:
                    json.dump(tag_files, file, ensure_ascii=False, indent=4)
    if signal_received:
        print(f"Received signal {signal_received}. Results saved to relevant_posts.json")
        exit(0)

# Signal handling
def signal_handler(sig, frame):
    save_results(tag_files, sig)

# Register signal handler
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def find_relevant_posts():
    """
    Find relevant Stack Overflow posts for each challenge based on keyword similarity.
    
    Args:
        posts: List of Stack Overflow posts
        challenges: List of tuples containing challenge name and keywords with cosine similarities
        min_similarity: Minimum average cosine similarity for a post to be considered relevant
    
    Returns:
        Dictionary mapping challenge names to lists of relevant posts
    """
    kw_model = KeyBERT(model='flax-sentence-embeddings/stackoverflow_mpnet-base')
    relevant_posts = {}

    for category in challenges:
        for subcategory in category['subcategories']:
            for challenge in subcategory['challenges']:
                challenge_id = challenge['query_id']
                challenge_name = challenge['name']
                keywords = challenge['keywords'] 
                threshold_cosine = challenge['average_similarity']
                if challenge_id < 8:
                    continue
                # traverse through each file at pathvquery_results/{keyword}/stackoverflow_results.json
                # for each file, check if the average cosine similarity is greater than the threshold
                # if it is greater, add the post to the list of relevant posts
                relevant_posts[challenge_id] = []
                tag_files = {}
                for keyword in keywords:
                    with open(f'query_results/{keyword}/stackoverflow_results.json', 'r', encoding='utf-8') as file:
                        posts = json.load(file)
                        candidate = []
                    for post in posts:
                        candidate.extend(post['tags'])
                        for tag in post['tags']:
                            if tag not in tag_files:
                                tag_files[tag] = {'posts': [], 'relevance': False, 'similar': False ,'cosine_similarity': 0.0}
                            tag_files[tag]['posts'].append(post)

                tag_candidate = tag_files.keys()
                keywords_list = kw_model.extract_keywords(f"{challenge['name']}: {challenge['description']}", candidates = tag_candidate, keyphrase_ngram_range=(1, 1))
                for keyword in keywords_list:
                    tag_files[keyword[0]]['cosine_similarity'] = keyword[1]
                    if keyword[1] > threshold_cosine:
                        tag_files[keyword[0]]['similar'] = True
                # save tags_file to a json file
                save_results(tag_files, challenge_id)
                

if __name__ == "__main__":
    # get_threshold(docs)
    find_relevant_posts()