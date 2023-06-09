import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import matplotlib.pyplot as plt
from langdetect import detect, DetectorFactory

# Read metadata.csv into a pandas DataFrame
df = pd.read_csv('metadata.csv')

# Use only a portion of the dataset (e.g., first 10,000 rows)
df = df.head(10000)

# Drop rows with NaN values in the 'abstract' column
df = df.dropna(subset=['abstract'])

# Replace cells with NaN values in the 'title' column with ''. We maintain these rows as documents' contents are a priority.
df['title'] = df['title'].fillna('')

DetectorFactory.seed = 0

# Function to detect the language of the text
def detect_language(text):
    try:
        return detect(text)
    except:
        return None

# Remove non-English rows
df['language'] = df['title'].apply(detect_language)
df = df[df['language'] == 'en']

# Concatenate the title and abstract columns to form a new column called 'title_abstract'
df['title_abstract'] = df['title'] + ' ' + df['abstract']

# Load a pre-trained BERT model
model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

# Generate sentence embeddings for each full text in the DataFrame
embeddings = model.encode(df['title_abstract'].tolist(), convert_to_tensor=True)

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
    
    # Compute the cosine similarity scores and take the ones corresponding to the top 10 results
    cos_scores = torch.nn.functional.cosine_similarity(model.encode(query, convert_to_tensor=True), embeddings).cpu().numpy()
    top_indices = cos_scores.argsort()[-top_n:][::-1]
    top_cos_scores = cos_scores[top_indices]
    print(f"\nThe scores array for the top ten results: {top_cos_scores}")

    # Print the top results to the console alongside the scores
    print(f"\nTop {top_n} results for the query '{query}':")
    for (index, row), score in zip(results.iterrows(), top_cos_scores):
        print(f"\nScore: {score}")
        print(f"\nTitle: {row['title']}\nAbstract: {row['abstract']}\nURL: {row['url']}")
        
    # Create a Pandas dataframe from the Titles and Scores for the top 10 results
    df_scores = pd.DataFrame({"Titles":results['title'], "Scores":top_cos_scores})
    df_scores_sorted = df_scores.sort_values('Scores')

    # Create a bar chart of the cosine similarity scores for the top 10 results
    plt.barh("Titles", "Scores", data = df_scores_sorted, height = 0.4)
    plt.title(f"Top 10 Cosine Similarity Scores for Query '{query}'")
    plt.xlabel("Cosine Similarity Score")
    plt.xticks() 
    plt.show()
