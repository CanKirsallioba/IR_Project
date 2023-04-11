# Search and Retrieval

This repository contains two Python scripts for search and retrieval:

    bm25.py: Uses BM25 to retrieve the most relevant results from a given query
    bert.py: Uses BERT to encode documents and retrieve the most similar results to a given query

## Dependencies

To run these scripts, you will need the following libraries:

    pandas
    nltk
    math
    matplotlib
    langdetect
    sentence_transformers
    torch
    numpy

You can install these dependencies by running the following command:

    pip install -r requirements.txt

## Usage

To use these scripts, you will need to provide the metadata.csv file which is on https://www.kaggle.com/datasets/allen-institute-for-ai/CORD-19-research-challenge .

To run the BM25 script, use the following command:

    python search_bm25.py

This will prompt you to enter a query string. The script will return the top 10 results that are most relevant to your query, along with their corresponding BM25 scores. It will also display a bar chart of the scores.

To run the BERT script, use the following command:

    python search_bert.py

This will prompt you to enter a query string. The script will return the top 10 results that are most similar to your query, along with their corresponding cosine similarity scores. It will also display a bar chart of the scores.
