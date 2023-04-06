import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
import numpy as np

# Read metadata.csv
df = pd.read_csv('metadata.csv')

# Use only a portion of the dataset (e.g., first 10,000 rows)
df = df.head(10000)

# Concatenate title and abstract columns
df['full_text'] = df['title'].fillna('') + ' ' + df['abstract'].fillna('')

# Load a pre-trained BERT model
model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

# Generate embeddings for the full_text column
embeddings = model.encode(df['full_text'].tolist(), convert_to_tensor=True)


def search(query, embeddings, top_n=10):
    query_embedding = model.encode(query, convert_to_tensor=True)
    cos_scores = torch.nn.functional.cosine_similarity(query_embedding, embeddings).cpu().numpy()
    top_indices = cos_scores.argsort()[-top_n:][::-1]
    return df.iloc[top_indices]


if __name__ == "__main__":
    query = "COVID-19 transmission"
    top_n = 10
    results = search(query, embeddings, top_n)

    print(f"Top {top_n} results for the query '{query}':")
    for index, row in results.iterrows():
        print(f"\nTitle: {row['title']}\nAbstract: {row['abstract']}\nURL: {row['url']}")
