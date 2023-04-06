import pandas as pd
import nltk
from rank_bm25 import BM25Okapi

nltk.download('punkt')

# Read metadata.csv
df = pd.read_csv('metadata.csv')

# Preprocess the text
def preprocess(text):
    return nltk.word_tokenize(text.lower())

# Tokenize the title and abstract columns
tokenized_corpus = []
for index, row in df.iterrows():
    if type(row['title']) == str:
        title_tokens = preprocess(row['title'])
    else:
        title_tokens = []

    if type(row['abstract']) == str:
        abstract_tokens = preprocess(row['abstract'])
    else:
        abstract_tokens = []

    tokenized_corpus.append(title_tokens + abstract_tokens)

# Initialize the BM25 model
bm25 = BM25Okapi(tokenized_corpus)


def search(query, top_n=10):
    query_tokens = preprocess(query)
    scores = bm25.get_scores(query_tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
    return df.iloc[top_indices]



if __name__ == "__main__":
    query = "COVID-19 transmission"
    top_n = 10
    results = search(query, top_n)

    print(f"Top {top_n} results for the query '{query}':")
    for index, row in results.iterrows():
        print(f"\nTitle: {row['title']}\nAbstract: {row['abstract']}\nURL: {row['url']}")
