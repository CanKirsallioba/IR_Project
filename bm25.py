import pandas as pd
import nltk
import math

nltk.download('punkt')

# Read metadata.csv and use only the first 10,000 rows
df = pd.read_csv('metadata.csv', nrows=10000)

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

# BM25 Implementation
def idf(token, tokenized_corpus):
    N = len(tokenized_corpus)
    n_q = sum(1 for doc in tokenized_corpus if token in doc)
    return math.log((N - n_q + 0.5) / (n_q + 0.5))

def bm25(query_tokens, doc_tokens, tokenized_corpus, k1=1.5, b=0.75):
    doc_len = len(doc_tokens)
    avg_doc_len = sum(len(doc) for doc in tokenized_corpus) / len(tokenized_corpus)
    score = 0

    for token in query_tokens:
        if token not in doc_tokens:
            continue

        f_q = doc_tokens.count(token)
        term_idf = idf(token, tokenized_corpus)
        numerator = f_q * (k1 + 1)
        denominator = f_q + k1 * (1 - b + b * (doc_len / avg_doc_len))
        score += term_idf * (numerator / denominator)

    return score

# Search function
def search(query, tokenized_corpus, top_n=10):
    query_tokens = preprocess(query)
    scores = [bm25(query_tokens, doc_tokens, tokenized_corpus) for doc_tokens in tokenized_corpus]
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
    return df.iloc[top_indices]

# Perform a query
if __name__ == "__main__":
    query = "COVID-19 transmission"
    top_n = 10
    results = search(query, tokenized_corpus, top_n)

    print(f"Top {top_n} results for the query '{query}':")
    for index, row in results.iterrows():
        print(f"\nTitle: {row['title']}\nAbstract: {row['abstract']}\nURL: {row['url']}")
