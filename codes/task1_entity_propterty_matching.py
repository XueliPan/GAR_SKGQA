""""
Task 1: Entity-Property Matching
This script matches entities and properties from a input question to the top-k similar entities and properties in the target KGs.
It uses the Qwen3-Embedding-0.6B model to compute embeddings and find the most similar candidates.
"""

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



def get_response(client, prompt, instruction):
    response = client.models.generate_content(
    model="gemini-2.5-flash-lite",
    config=types.GenerateContentConfig(
        system_instruction=instruction),
    contents=prompt
)
    return response.text


# Define schema with Pydantic
class ExtractionResult(BaseModel):
    Entities: List[str]
    Properties: List[str]

# Example LLM response (your code will replace this with the actual response)
# response = get_response(prompt, instruction)

def validate_response(response: str) -> bool:
    try:
        # Parse string to dict
        data = json.loads(response)

        # Validate with Pydantic
        ExtractionResult(**data)
        return True  # ✅ Valid JSON + matches schema
    
    except (json.JSONDecodeError, ValidationError):
        return False  # ❌ Invalid JSON or schema mismatch
    

def get_top_k_candidates(sources:list, candidates:list, top_k=5):
    """
    Match source entities/properties to the top-k similar candidates using Qwen3-Embedding-0.6B model
    :param sources: list of source entities/properties
    :param candidates: list of candidate entities/properties
    :param top_k: number of top similar candidates to return
    :return: DataFrame with source, candidate, and similarity scores
    """
    model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
    candidate_embeddings = model.encode(candidates)
    # Encode the queries and documents. Note that queries benefit from using a prompt
    # Here we use the prompt called "query" stored under `model.prompts`, but you can
    # also pass your own prompt via the `prompt` argument
    source_embeddings = model.encode(sources, prompt_name="query")
    # Compute the (cosine) similarity between the source and candidates embeddings
    similarity = model.similarity(source_embeddings, candidate_embeddings)
    # print the top 5 most similar candidate properties for each source property
    similarity = torch.tensor(similarity)
    top_k = 5
    values, indices = torch.topk(similarity, k=top_k, dim=1)
    top_candidates = []
    top_scores = []
    top_sources = []
    for i, source_property in enumerate(sources):
        # print(f"Source property: {source_property}")
        for j in range(top_k):
            candidate_property = candidates[indices[i][j]]
            sim_score = values[i][j].item()
            # print(f"  Candidate property: {candidate_property}, similarity score: {sim_score:.4f}")
            top_candidates.append(candidate_property)
            top_scores.append(f"{sim_score:.4f}")
            top_sources.append(source_property)
    top_df = pd.DataFrame({
        "source_entity_property": top_sources,
        "candidate_entity_property": top_candidates,
        "similarity_score": top_scores
    })
    return top_df


def main(input_question):
    #--------------entity or property recognition----------------#
    # Initialize the google GenAI client
    # load genai API key from .env file
    load_dotenv()
    client = genai.Client()
    instruction = """
        You are an expert in entity recognition and relation extraction. 
        You will be given a question and you need to identify and extract all the entities and properties mentioned in the question. 
        Please provide a list of entities and properties found in the following JSON format. Make sure to use double quotes for both the keys and string values. Make sure to use the raw json output instead of a markdown code block.

        {
        "Entities": ["entity1", "entity2", "..."],
        "Properties": ["property1", "property2", "..."]
        }

        If no entities or properties are found, please return empty lists.
        """
    response = get_response(client, input_question, instruction)
    print("Raw LLM output:", response)
    print("Is response valid?", validate_response(response))
    if not validate_response(response):
        print("LLM output is not valid. Please check the response format.")
    else:
        pass
    #--------------entity or property matching----------------#
    # match entities and properties to the top-k similar candidates in the target KGs
    property_df = pd.read_csv("/Users/sherrypan/GitHub/GAR_SKGQA/datasets/sciqa/project_data/property-labels.csv")
    candidate_property_ls = property_df["label"].tolist()

    entity_df = pd.read_csv("/Users/sherrypan/GitHub/GAR_SKGQA/datasets/sciqa/project_data/subject-labels.csv")
    candidate_entity_ls = entity_df["label"].tolist()

    # for the json output from LLM, we can use json.loads() to convert it to a a list of entities and properties
    response_dict = json.loads(response)
    source_entities = response_dict.get("Entities", [])
    source_properties = response_dict.get("Properties", [])

    # Match entities and properties to the top-k similar candidates
    entity_matches = get_top_k_candidates(source_entities, candidate_entity_ls)
    property_matches = get_top_k_candidates(source_properties, candidate_property_ls)

    print("Entity Matches:")
    print(entity_matches)
    print("Property Matches:")
    print(property_matches)

if __name__ == "__main__":
    input_question = "What data format does CHEMDNER corpus have?"
    main(input_question)



