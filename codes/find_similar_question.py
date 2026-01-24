import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm

# Load CSV files
HQ_df = pd.read_csv("datasets/sciqa/project_data/sciqa-all-HQs.csv")
AQ_df = pd.read_csv("datasets/sciqa/project_data/sciqa-all-AQs.csv")

# Load Sentence-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')  # Small, fast, and good quality

# Encode all questions using Sentence-BERT
print("Encoding train questions...")
targe_embedding = model.encode(AQ_df["question"].tolist(), convert_to_tensor=True)

print("Encoding test questions and finding most similar train question...")
results = []

# Iterate through each test question
for idx, row in tqdm(HQ_df.iterrows(), total=len(HQ_df)):
    HQ_id = row["id"]
    HQ_question = row["question"]
    HQ_query = row["query"]

    # Encode test question
    source_embedding = model.encode(HQ_question, convert_to_tensor=True)

    # Compute cosine similarities
    cosine_scores = cosine_similarity([source_embedding.cpu().numpy()], targe_embedding.cpu().numpy())[0]

    # Find index of best match
    best_match_idx = np.argmax(cosine_scores)

    # Get best matched train question and query
    best_AQ_question = AQ_df.iloc[best_match_idx]["question"]
    best_AQ_query = AQ_df.iloc[best_match_idx]["query"]

    results.append({
        "HQ_id": HQ_id,
        "HQ_string": HQ_question,
        "HQ_query": HQ_query,
        "similar_AQ_question": best_AQ_question,
        "similar_AQ_query": best_AQ_query
    })

# Save results to CSV
output_df = pd.DataFrame(results)
output_df.to_csv("datasets/sciqa/project_data/oneshot4HQs.csv", index=False)
print("Saved to oneshot4HQs.csv")