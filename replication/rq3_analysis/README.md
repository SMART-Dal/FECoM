# RQ3 Analysis

This README file provides an overview of the RQ3 analysis conducted in the study "Enhancing Energy-Awareness in Deep Learning through Fine-Grained Energy Measurement." It also serves as a guide for replicating the analysis.

## Overview

RQ3 explores the key considerations and challenges faced when designing and developing frameworks and tools similar to FECoM for fine-grained energy consumption measurement. The analysis follows a multi-step approach that involves:

1. Documenting and categorizing challenges encountered during the development of FECoM.
2. Extracting semantically similar keyword queries for each challenge using the Claude-3 Opus Language Model (LLM) API.
3. Searching for relevant Stack Overflow posts using the extracted keyword queries.
4. Filtering the retrieved posts using an Information Retrieval-based search pipeline.
5. Manually analyzing the filtered posts to identify relevant challenges and mitigation strategies.

## Replication Steps

To replicate the RQ3 analysis, follow these steps:

1. **Keyword Extraction**:
   - Use the Claude-3 Opus LLM API or any other capable LLM to extract semantically similar keyword queries for each challenge listed in `replication/rq3_analysis/rq3_data.py`. In this study, for each challange we repeat the process for 10 times,and consider most frequent queries.
   - The prompt for query generation can be found in `replication/rq3_analysis/prompt.txt`. Results are stored in the `replication/rq3_analysis/keywords_results`.

2. **Corpus Construction**:
   - Use the extracted keyword queries to search for relevant Stack Overflow posts using the Stack Overflow API provided through the Stack Exchange platform.
   - The script `replication/rq3_analysis/stxchng_search.py` demonstrates how to use the Stack Overflow API to mine posts.
   - Customize the script with your own access token and key.
   - The script will save the retrieved posts in JSON format within the `query_results` folder.

3. **Corpus Filtering**:
   - Create an Information Retrieval-based search pipeline to filter out irrelevant posts.
   - The script `replication/rq3_analysis/filtered_posts/filter_posts.py` is used to create the search index and IR model.
   - The script combines the challenge descriptions with the Stack Overflow posts using ColBERT, a state-of-the-art retrieval model.
   - It constructs a search index of the Stack Overflow posts and retrieves the top 50 most relevant posts for each challenge.

4. **Manual Analysis**:
   - Manually analyze the filtered corpus of Stack Overflow posts to identify the ones discussing challenges and potential mitigation strategies relevant to the context.
   - Review multiple posts for each challenge, extract relevant mitigation strategies, and summarize them.
   - The mapping between the specific posts used for this analysis and the corresponding identified challenges can be found in `c1/`, `c2/`, ..., `c12/` in `filtered_posts`.

## Folder Structure

- `replication/rq3_analysis/`: Contains all the files and scripts related to the RQ3 analysis.
  - `stxchng_search.py`: Script to mine Stack Overflow posts using the Stack Overflow API.
  - `prompt.txt`: Contains the prompt for query generation.
  - `create_folder.py`: Script to create folders for mining posts for keywords.
  - `rq3_data.py`: Contains JSON data with all challenges information.
  - `filtered_posts/`: Contains files related to corpus filtering.
    - `filter_posts.py`: Script to create the search index and IR model.
  - `c1/`, `c2/`, ..., `c12/`: Folders corresponding to each challenge, containing ranked results of documents returned by the IR model, along with the mapped documents used during manual analysis for each challanges mitigation strategy.

## References

- Claude-3 Opus LLM API: [https://www.anthropic.com](https://www.anthropic.com)
- Stack Exchange API: [https://api.stackexchange.com](https://api.stackexchange.com)
- ColBERT: [https://github.com/stanford-futuredata/ColBERT](https://github.com/stanford-futuredata/ColBERT)
- RAGatouille [https://github.com/bclavie/RAGatouille/tree/main](https://github.com/bclavie/RAGatouille/tree/main)
