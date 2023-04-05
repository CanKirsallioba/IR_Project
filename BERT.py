import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load a small subset of the metadata.csv dataset
df = pd.read_csv('metadata.csv', nrows=100)


# Initialize the BERT-based model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Convert the documents to sequences of tokens and encode them
sequences = df['abstract'].fillna('').tolist()
encoded_sequences = tokenizer.batch_encode_plus(
    sequences,
    padding=True,
    truncation=True,
    return_tensors='pt'
)

# Pass the encoded sequences through the model to obtain document scores
outputs = model(**encoded_sequences)
scores = outputs.logits.squeeze()

# Sort the documents by their scores in descending order
indices = torch.argsort(scores, descending=True)
ranked_documents = df.iloc[indices.numpy()]

# Print the top 10 ranked documents
print(ranked_documents.head(10))




