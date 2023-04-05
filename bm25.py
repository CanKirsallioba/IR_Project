import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from tqdm import tqdm

# Load the metadata.csv file using pandas
metadata = pd.read_csv("metadata.csv", dtype={"abstract": str}, low_memory=False)

# Define the BM25 parameters
k1 = 1.2
b = 0.75

# Preprocess the text data
def preprocess_text(text):
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        # Remove punctuations
        text = re.sub(r'[^\w\s]','',text)
        # Tokenize the text
        tokens = nltk.word_tokenize(text)
        # Remove stop words
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        # Stem the words
        stemmer = SnowballStemmer("english")
        tokens = [stemmer.stem(token) for token in tqdm(tokens)]
        # Join the tokens back into a string
        text = " ".join(tokens)
    else:
        text = ""
    return text

metadata['processed_text'] = metadata['abstract'].apply(preprocess_text)

print('After preprocessing ...')

# Create the document-term matrix
count_vect = CountVectorizer()
doc_term_matrix = count_vect.fit_transform(metadata['processed_text'])

# Compute the IDF values
tfidf_transformer = TfidfTransformer()
tfidf_transformer.fit(doc_term_matrix)
idf = tfidf_transformer.idf_

# Compute the document length
doc_len = np.sum(doc_term_matrix, axis=1)

# Compute the average document length
avg_doc_len = np.mean(doc_len)

# Compute the BM25 score for each document
def compute_bm25_score(query, doc_idx):
    # Get the document length
    dl = doc_len[doc_idx]
    # Convert the query to a document-term matrix
    query_matrix = count_vect.transform([query])
    # Compute the term frequency in the query
    tf = query_matrix.sum(axis=1)
    # Compute the document frequency for each term in the query
    df = doc_term_matrix[:, query_matrix.indices].sum(axis=0)
    # Compute the BM25 score for each term in the query
    score = np.sum(np.log((metadata.shape[0] - df + 0.5) / (df + 0.5)) * ((k1 + 1) * tf) / (k1 * ((1 - b) + b * (dl / avg_doc_len)) + tf))
    return score

# Define the query
query = "COVID-19 pandemic"

# Compute the BM25 score for each document
bm25_scores = []
for i in tqdm(range(metadata.shape[0])):
    score = compute_bm25_score(query, i)
    bm25_scores.append(score)

# Sort the documents based on their BM25 scores
idx_sorted = np.argsort(bm25_scores)[::-1]

# Print the top-k documents with the highest BM25 scores
k = 10
for i in range(k):
    idx = idx_sorted[i]
    print("Title:", metadata.loc[idx, "title"])
    print("Authors:", metadata.loc[idx, "authors"])
    print("BM25 score:", bm25_scores[idx])
    print("Abstract:", metadata.loc[idx, "abstract"])
    print("\n")
