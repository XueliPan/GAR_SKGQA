import time
import json
import pandas as pd
import os
import numpy as np
import torch
from pydantic import BaseModel, ValidationError
from typing import List, Optional
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
from helper import run_sparql_query

# Optional imports for LLM backends
try:
    from google import genai  # type: ignore
    from google.genai import types  # type: ignore
except Exception:  # pragma: no cover
    genai = None
    types = None

try:
    from transformers import pipeline  # type: ignore
except Exception:  # pragma: no cover
    pipeline = None

# --- New Caching Section ---
# Pre-computation and storage of candidate embeddings
# This part runs only once when the script starts or when the cache is not found.

# File paths for caching
ENTITY_EMBEDDINGS_PATH = "entity_embeddings.npy"
PROPERTY_EMBEDDINGS_PATH = "property_embeddings.npy"
CANDIDATE_ENTITIES_PATH = "datasets/sciqa/project_data/orkg_entity_labels.csv"
CANDIDATE_PROPERTIES_PATH = "datasets/sciqa/project_data/sciqa_predicate_labels.csv"

# Load the model only once
embedding_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

def load_or_compute_embeddings(csv_path: str, embeddings_path: str):
    """Loads embeddings from cache or computes and saves them.
    Ensures labels are strings and drops missing values.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if "label" not in df.columns:
        raise KeyError(f"'label' column not found in {csv_path}. Columns: {df.columns.tolist()}")

    # Drop rows where label is missing, then convert all labels to str and strip whitespace
    df = df[df["label"].notna()].copy()
    df["label"] = df["label"].astype(str).str.strip()

    candidates = df["label"].tolist()

    if os.path.exists(embeddings_path):
        print(f"Loading embeddings from cache: {embeddings_path}")
        embeddings = np.load(embeddings_path)
        # ensure shape matches
        if embeddings.shape[0] != len(candidates):
            print("Warning: cached embeddings length differs from candidates; recomputing.")
        else:
            return candidates, embeddings

    # compute embeddings and save
    print(f"Computing embeddings for {len(candidates)} candidates and saving to {embeddings_path}")
    embeddings = embedding_model.encode(candidates, convert_to_numpy=True, show_progress_bar=True, batch_size=64)
    # ensure float32 for downstream torch conversion
    embeddings = embeddings.astype(np.float32)
    np.save(embeddings_path, embeddings)
    return candidates, embeddings

# --- End of New Caching Section ---

def get_response(prompt: str, instruction: str, backend: str = "google", model: Optional[str] = None, temperature: float = 0.1) -> str:
    """
    Generate a response using the selected backend (google or transformers).
    For google, uses system_instruction + contents; for transformers, concatenates instruction and prompt.
    Includes simple retry on transient overload for google.
    """
    backend = (backend or "google").lower()

    if backend == "google":
        if genai is None or types is None:
            raise RuntimeError("google-genai is not installed. Install google-genai to use backend=google.")
        overload_markers = [
            "503", "UNAVAILABLE", "overloaded", "The model is overloaded", "Rate limit", "try again later",
        ]
        model_id = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        while True:
            try:
                client = genai.Client()
                resp = client.models.generate_content(
                    model=model_id,
                    config=types.GenerateContentConfig(system_instruction=instruction, temperature=float(temperature)),
                    contents=prompt,
                )
                return resp.text
            except Exception as e:
                msg = str(e)
                if any(m in msg for m in overload_markers):
                    print("Gemini overloaded/unavailable. Waiting 30s before retry...")
                    time.sleep(30)
                    continue
                raise

    elif backend == "transformers":
        if pipeline is None:
            raise RuntimeError("transformers is not installed. Install transformers to use backend=transformers.")
        model_id = model or os.getenv("HF_TEXTGEN_MODEL", "meta-llama/Llama-3.1-8B")
        textgen = pipeline("text-generation", model=model_id, dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
        combined = f"System Instruction:\n{instruction}\n\nUser:\n{prompt}"
        outputs = textgen(combined, max_new_tokens=512, do_sample=False, temperature=float(temperature))
        generated = outputs[0].get("generated_text", "").strip()
        if generated.startswith(combined):
            generated = generated[len(combined):].strip()
        return generated

    else:
        raise ValueError(f"Unsupported backend: {backend}. Use 'google' or 'transformers'.")

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


def get_top_k_candidates(sources: list, candidates: list, candidate_embeddings: np.ndarray, top_k=3) -> pd.DataFrame:
    """
    Match source entities/properties to top-k similar candidates using pre-computed embeddings.
    Uses sentence_transformers.util.cos_sim for stable cosine computation.
    """
    if not sources:
        return pd.DataFrame({
            "source_entity_property": [],
            "candidate_entity_property": [],
            "similarity_score": []
        })

    # Ensure inputs are strings and drop empty source strings
    cleaned_sources = [str(s).strip() for s in sources if s is not None and str(s).strip() != ""]
    if not cleaned_sources:
        return pd.DataFrame({
            "source_entity_property": [],
            "candidate_entity_property": [],
            "similarity_score": []
        })

    # Encode sources (returns numpy by default as convert_to_numpy=True)
    source_embeddings = embedding_model.encode(cleaned_sources, convert_to_numpy=True)

    # Convert numpy embeddings to torch tensors for util.cos_sim
    src_tensor = torch.from_numpy(source_embeddings) if isinstance(source_embeddings, np.ndarray) else torch.tensor(source_embeddings)
    cand_tensor = torch.from_numpy(candidate_embeddings) if isinstance(candidate_embeddings, np.ndarray) else torch.tensor(candidate_embeddings)

    # Compute cosine similarity matrix: (len(sources), len(candidates))
    similarity = util.cos_sim(src_tensor, cand_tensor)  # returns a torch.Tensor

    # Get top-k indices and values per source row
    top_k = min(top_k, similarity.size(1))
    values, indices = torch.topk(similarity, k=top_k, dim=1)

    top_candidates = []
    top_scores = []
    top_sources = []

    for i, src in enumerate(cleaned_sources):
        for j in range(top_k):
            idx = indices[i][j].item()
            candidate_property = candidates[idx]
            sim_score = float(values[i][j].item())
            top_candidates.append(candidate_property)
            top_scores.append(f"{sim_score:.4f}")
            top_sources.append(src)

    top_df = pd.DataFrame({
        "source_entity_property": top_sources,
        "candidate_entity_property": top_candidates,
        "similarity_score": top_scores
    })
    return top_df

# Main function, modified to pass pre-computed embeddings
def main(input_question: str, backend: str = "google", model: Optional[str] = None, temperature: float = 0.1):
    # Load genai API key from .env file
    load_dotenv()
    
    # Print device information for debugging
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"\nCUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name()}")
    
    instruction = """
        You are an expert in entity recognition and predicate extraction for scholarly data mining. 
        You will be given a question and you need to identify and extract all the entities and predicates mentioned in the question. 
        Entities could be the specific names of papers, authors, datasets, models, methods, benchmarks, etc in research papers. Not general terms such as ID, title, study, paper, model, dataset etc.
        Predicates are the relationships or attributes associated with these entities.
        Your response must be a single, raw JSON string. Do not include any text before or after the JSON.  

        Please provide a list of entities and predicates found in the following format:

        {
        "Entities": ["entity1", "entity2", "..."],
        "Predicates": ["predicate1", "predicate2", "..."]
        }

        If no entities or predicates are found, please return empty lists within the JSON object.
        Do not wrap the JSON in any markdown code block, such as ```json or ```.
        """
    response = get_response(input_question, instruction, backend=backend, model=model, temperature=temperature)
    print(f"\nRaw LLM output:\n{response}\n")
    # print("Is response valid?", validate_response(response))

    # if the output contain extra text, try to extract the JSON part
    if not validate_response(response):
        try:
            start = response.index("{")
            end = response.rindex("}") + 1
            response = response[start:end]
            print(f"Extracted JSON part:\n{response}\n")
        except ValueError:
            print("Could not find JSON object in the response.")
            return

    response_dict = json.loads(response)
    source_entities = response_dict.get("Entities", [])
    source_properties = response_dict.get("Predicates", [])
    
    # Pass the pre-computed embeddings to the matching function
    candidate_entity_ls, candidate_entity_embeddings = load_or_compute_embeddings(CANDIDATE_ENTITIES_PATH, ENTITY_EMBEDDINGS_PATH)
    print("Loaded candidate entities and embeddings:", len(candidate_entity_ls))
    candidate_property_ls, candidate_property_embeddings = load_or_compute_embeddings(CANDIDATE_PROPERTIES_PATH, PROPERTY_EMBEDDINGS_PATH)
    print("Loaded candidate properties and embeddings:", len(candidate_property_ls))

    entity_matches = get_top_k_candidates(source_entities, candidate_entity_ls, candidate_entity_embeddings)
    property_matches = get_top_k_candidates(source_properties, candidate_property_ls, candidate_property_embeddings)

    print("\n----- Matching Results via Vector Similarity -----")
    print("\nEntity Matches for:", input_question)
    print(entity_matches)
    print("\nPredicate Matches for:", input_question)
    print(property_matches)
    return entity_matches, property_matches

if __name__ == "__main__":
    input_questions = [
    "What data format does CHEMDNER corpus have?",
    "Where did the study with maximal geographic scale take place?",
    "What quantity of iron oxide was discovered on Elorza crater?",
    "What types of nanocarriers do have therapeutic effect?",
    "What models are being evaluated on the TDMSci dataset?",
    "List the title and ID of research papers that contain a benchmark over the Penn Treebank (Word Level) dataset?",
    "Provide a list of papers that have utilized the Flair-TDM model and include the links to their code?",
    "Where can I find code references in papers that have used the CATTS-XSUM model for benchmarking purposes?"
    ]
    for input_question in input_questions:
        main(input_question)
        print("\n"+"="*100+"\n")