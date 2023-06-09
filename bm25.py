import pandas as pd
import nltk
import math
import matplotlib.pyplot as plt
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0

# Download the necessary NLTK resources
nltk.download('punkt')

# Read metadata.csv and use only the first 10,000 rows
df = pd.read_csv('metadata.csv', nrows=10000)

# Function to detect the language of the text
def detect_language(text):
    try:
        return detect(text)
    except:
        return None

# Remove non-English rows
df['language'] = df['title'].apply(detect_language)
df = df[df['language'] == 'en']

# Define a function for preprocessing text
def preprocess(text):
    # Tokenize the text into words and convert to lowercase
    return nltk.word_tokenize(text.lower())

# Tokenize the title and abstract columns of the DataFrame
tokenized_corpus = []
for index, row in df.iterrows():
    # Tokenize the title column if it is a string, otherwise use an empty list
    if type(row['title']) == str:
        title_tokens = preprocess(row['title'])
    else:
        title_tokens = []

    # Tokenize the abstract column if it is a string, otherwise use an empty list
    if type(row['abstract']) == str:
        abstract_tokens = preprocess(row['abstract'])
    else:
        abstract_tokens = []

    # Combine the title and abstract tokens into a single document
    tokenized_corpus.append(title_tokens + abstract_tokens)

# Define a function for computing the inverse document frequency of a term
def idf(token, tokenized_corpus):
    N = len(tokenized_corpus)
    n_q = sum(1 for doc in tokenized_corpus if token in doc)
    return math.log((N - n_q + 0.5) / (n_q + 0.5))

# Define a function for computing the BM25 score between a query and a document
def bm25(query_tokens, doc_tokens, tokenized_corpus, k1=1.5, b=0.75):
    doc_len = len(doc_tokens)
    avg_doc_len = sum(len(doc) for doc in tokenized_corpus) / len(tokenized_corpus)
    score = 0

    for token in query_tokens:
        # Skip the token if it is not present in the document
        if token not in doc_tokens:
            continue

        f_q = doc_tokens.count(token)
        term_idf = idf(token, tokenized_corpus)
        numerator = f_q * (k1 + 1)
        denominator = f_q + k1 * (1 - b + b * (doc_len / avg_doc_len))
        score += term_idf * (numerator / denominator)

    return score

# Define a function for searching the DataFrame using a query string
def search(query, tokenized_corpus, top_n=10):
    # Preprocess the query string into tokens
    query_tokens = preprocess(query)

    # Compute the BM25 scores between the query and each document in the corpus
    scores = [bm25(query_tokens, doc_tokens, tokenized_corpus) for doc_tokens in tokenized_corpus]

    # Get the indices of the top n scores in descending order
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]

    # Get the BM25 scores of the top n documents
    top_scores = [scores[i] for i in top_indices]

    # Return the DataFrame rows and scores corresponding to the top n scores
    return df.iloc[top_indices], top_scores


if __name__ == "__main__":
    query = input("Please enter your query: ")
    top_n = 10
    results, scores = search(query, tokenized_corpus, top_n)

    # Print the top results and scores to the console
    print(f"Top {top_n} results for the query '{query}':")
    for result_index, (index, row) in enumerate(results.iterrows()):
        print(f"\nTitle: {row['title']}\nAbstract: {row['abstract']}\nURL: {row['url']}\nScore: {scores[result_index]}")

    # Create a bar chart of the BM25 scores
    fig, ax = plt.subplots()
    ax.bar(range(len(scores)), scores)
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels([row['title'] for index, row in results.iterrows()], rotation=90)
    ax.set_title(f"Top {top_n} Results for the Query '{query}'")
    ax.set_xlabel("Document")
    ax.set_ylabel("BM25 Score")
    plt.show()

