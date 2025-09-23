import time
from urllib import response
import torch
import json
import pandas as pd
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer
from helper import run_sparql_query
from pydantic import BaseModel, ValidationError
from typing import List
from dotenv import load_dotenv
import numpy as np
import os

# --- New Caching Section ---
# Pre-computation and storage of candidate embeddings
# This part runs only once when the script starts or when the cache is not found.

# File paths for caching
ENTITY_EMBEDDINGS_PATH = "entity_embeddings.npy"
PROPERTY_EMBEDDINGS_PATH = "property_embeddings.npy"
CANDIDATE_ENTITIES_PATH = "/Users/sherrypan/GitHub/GAR_SKGQA/datasets/sciqa/project_data/subject-labels.csv"
CANDIDATE_PROPERTIES_PATH = "/Users/sherrypan/GitHub/GAR_SKGQA/datasets/sciqa/project_data/property-labels.csv"

# Load the model only once
model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

def load_or_compute_embeddings(csv_path: str, embeddings_path: str):
    """Loads embeddings from cache or computes and saves them."""
    if os.path.exists(embeddings_path):
        print(f"Loading embeddings from cache: {embeddings_path}")
        df = pd.read_csv(csv_path)
        candidates = df["label"].tolist()
        embeddings = np.load(embeddings_path)
        return candidates, embeddings
    else:
        print(f"Cache not found. Computing and saving embeddings to {embeddings_path}")
        df = pd.read_csv(csv_path)
        candidates = df["label"].tolist()
        embeddings = model.encode(candidates, convert_to_numpy=True)
        np.save(embeddings_path, embeddings)
        return candidates, embeddings

candidate_entity_ls, candidate_entity_embeddings = load_or_compute_embeddings(CANDIDATE_ENTITIES_PATH, ENTITY_EMBEDDINGS_PATH)
candidate_property_ls, candidate_property_embeddings = load_or_compute_embeddings(CANDIDATE_PROPERTIES_PATH, PROPERTY_EMBEDDINGS_PATH)

# --- End of New Caching Section ---

# Existing functions (unmodified)
def get_response(client, prompt, instruction):
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        config=types.GenerateContentConfig(
            system_instruction=instruction),
        contents=prompt
    )
    return response.text

class ExtractionResult(BaseModel):
    Entities: List[str]
    Properties: List[str]

def validate_response(response: str) -> bool:
    try:
        data = json.loads(response)
        ExtractionResult(**data)
        return True
    except (json.JSONDecodeError, ValidationError):
        return False

# Modified get_top_k_candidates function
def get_top_k_candidates(sources:list, candidates:list, candidate_embeddings: np.ndarray, top_k=5):
    """
    Match source entities/properties to top-k similar candidates using pre-computed embeddings.
    """
    if not sources:
        return pd.DataFrame({
            "source_entity_property": [],
            "candidate_entity_property": [],
            "similarity_score": []
        })

    # Encode the new source queries only
    source_embeddings = model.encode(sources, prompt_name="query", convert_to_numpy=True)

    # Compute the (cosine) similarity
    similarity = model.similarity(source_embeddings, candidate_embeddings)
    similarity = torch.tensor(similarity)
    
    values, indices = torch.topk(similarity, k=top_k, dim=1)
    
    top_candidates = []
    top_scores = []
    top_sources = []
    for i, source_property in enumerate(sources):
        for j in range(top_k):
            candidate_property = candidates[indices[i][j]]
            sim_score = values[i][j].item()
            top_candidates.append(candidate_property)
            top_scores.append(f"{sim_score:.4f}")
            top_sources.append(source_property)

    top_df = pd.DataFrame({
        "source_entity_property": top_sources,
        "candidate_entity_property": top_candidates,
        "similarity_score": top_scores
    })
    return top_df

# Main function, modified to pass pre-computed embeddings
def main(input_question):
    # Load genai API key from .env file
    load_dotenv()
    client = genai.Client()
    instruction = """
        You are an expert in entity recognition and relation extraction. 
        You will be given a question and you need to identify and extract all the entities and properties mentioned in the question. 
        Your response must be a single, raw JSON string. Do not include any text before or after the JSON. Do not wrap the JSON in any markdown code block, such as ```json or ```. 

        Please provide a list of entities and properties found in the following format:

        {
        "Entities": ["entity1", "entity2", "..."],
        "Properties": ["property1", "property2", "..."]
        }

        If no entities or properties are found, please return empty lists within the JSON object.
        """
    response = get_response(client, input_question, instruction)
    print("Raw LLM output:", response)
    print("Is response valid?", validate_response(response))

    if not validate_response(response):
        print("LLM output is not valid. Please check the response format.")
        return

    response_dict = json.loads(response)
    source_entities = response_dict.get("Entities", [])
    source_properties = response_dict.get("Properties", [])
    
    # Pass the pre-computed embeddings to the matching function
    entity_matches = get_top_k_candidates(source_entities, candidate_entity_ls, candidate_entity_embeddings)
    property_matches = get_top_k_candidates(source_properties, candidate_property_ls, candidate_property_embeddings)

    print("----- Matching Results via Vector Similarity -----")
    print("Entity Matches:")
    print(entity_matches)
    print("\nProperty Matches:")
    print(property_matches)

if __name__ == "__main__":
    input_question = "What data format does CHEMDNER corpus have?"
    # input_question = "Where did the study with maximal geographic scale take place?"
    main(input_question)