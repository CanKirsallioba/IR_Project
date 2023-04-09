import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
import numpy as np

# Read metadata.csv into a pandas DataFrame
df = pd.read_csv('metadata.csv')

# Use only a portion of the dataset (e.g., first 10,000 rows)
df = df.head(10000)

# Concatenate the title and abstract columns to form a new column called 'full_text'
df['full_text'] = df['title'].fillna('') + ' ' + df['abstract'].fillna('')

# Load a pre-trained BERT model
model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

# Generate sentence embeddings for each full text in the DataFrame
embeddings = model.encode(df['full_text'].tolist(), convert_to_tensor=True)

# Define a function for searching the DataFrame using a query string
def search(query, embeddings, top_n=10):
    # Encode the query string to generate a query embedding
    query_embedding = model.encode(query, convert_to_tensor=True)

    # Compute the cosine similarity between the query embedding and the embeddings of each full text
    cos_scores = torch.nn.functional.cosine_similarity(query_embedding, embeddings).cpu().numpy()

    # Get the indices of the top n scores in descending order
    top_indices = cos_scores.argsort()[-top_n:][::-1]

    # Return the DataFrame rows corresponding to the top n scores
    return df.iloc[top_indices]

if __name__ == "__main__":
    # Prompt the user to enter a query string
    query = input("Please enter your query: ")
    
    # Set the number of top results to display
    top_n = 10
    
    # Call the search function to find the top results
    results = search(query, embeddings, top_n)

    # Print the top results to the console
    print(f"Top {top_n} results for the query '{query}':")
    for index, row in results.iterrows():
        print(f"\nTitle: {row['title']}\nAbstract: {row['abstract']}\nURL: {row['url']}")
