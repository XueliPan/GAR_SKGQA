import json
import os
import argparse
import time
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from pydantic import BaseModel, ValidationError
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM


# -------------------------- Config & Paths --------------------------

ENTITY_EMBEDDINGS_PATH = "entity_embeddings.npy"
PROPERTY_EMBEDDINGS_PATH = "property_embeddings.npy"
CANDIDATE_ENTITIES_PATH = "/Users/sherrypan/GitHub/GAR_SKGQA/datasets/sciqa/project_data/orkg_entity_labels.csv"
CANDIDATE_PROPERTIES_PATH = "/Users/sherrypan/GitHub/GAR_SKGQA/datasets/sciqa/project_data/sciqa_predicate_labels.csv"

# Use the same embedding model used elsewhere for consistency
embedding_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")


# -------------------------- Data Models ----------------------------

class ExtractionResult(BaseModel):
    Entities: List[str]
    Properties: List[str]


# -------------------------- Utilities ------------------------------

def load_or_compute_embeddings(csv_path: str, embeddings_path: str) -> Tuple[List[str], np.ndarray]:
    """Load cached embeddings if available, otherwise compute and cache them."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "label" not in df.columns:
        raise KeyError(f"'label' column not found in {csv_path}. Columns: {df.columns.tolist()}")

    df = df[df["label"].notna()].copy()
    df["label"] = df["label"].astype(str).str.strip()
    candidates = df["label"].tolist()

    if os.path.exists(embeddings_path):
        print(f"Loading embeddings from cache: {embeddings_path}")
        embeddings = np.load(embeddings_path)
        if embeddings.shape[0] == len(candidates):
            return candidates, embeddings
        print("Warning: cached embeddings length differs from candidates; recomputing.")

    print(f"Computing embeddings for {len(candidates)} candidates and saving to {embeddings_path}")
    embeddings = embedding_model.encode(candidates, convert_to_numpy=True, show_progress_bar=True, batch_size=64)
    embeddings = embeddings.astype(np.float32)
    np.save(embeddings_path, embeddings)
    return candidates, embeddings


def get_top_k_candidates(sources: List[str], candidates: List[str], candidate_embeddings: np.ndarray, top_k: int = 3) -> pd.DataFrame:
    """Compute top-k cosine similarity matches from sources to candidates."""
    if not sources:
        return pd.DataFrame({
            "source_entity_property": [],
            "candidate_entity_property": [],
            "similarity_score": []
        })

    cleaned_sources = [str(s).strip() for s in sources if s is not None and str(s).strip() != ""]
    if not cleaned_sources:
        return pd.DataFrame({
            "source_entity_property": [],
            "candidate_entity_property": [],
            "similarity_score": []
        })

    source_embeddings = embedding_model.encode(cleaned_sources, convert_to_numpy=True)
    src_tensor = torch.from_numpy(source_embeddings) if isinstance(source_embeddings, np.ndarray) else torch.tensor(source_embeddings)
    cand_tensor = torch.from_numpy(candidate_embeddings) if isinstance(candidate_embeddings, np.ndarray) else torch.tensor(candidate_embeddings)

    similarity = util.cos_sim(src_tensor, cand_tensor)
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

    return pd.DataFrame({
        "source_entity_property": top_sources,
        "candidate_entity_property": top_candidates,
        "similarity_score": top_scores
    })


def build_instruction() -> str:
    return (
        "You are an expert in entity recognition and predicate extraction for scholarly data mining. "
        "You will be given a question and you need to identify and extract all the entities and predicates mentioned in the question. "
        "Entities could be the specific names of papers, authors, datasets, models, methods, benchmarks, etc in research papers. "
        "Not general terms such as ID, title, study, paper, model, dataset etc. "
        "Predicates are the relationships or attributes associated with these entities.\n\n"
        "Return ONLY a single raw JSON string in this exact format (no extra text):\n"
        "{\n"
        "\"Entities\": [\"entity1\", \"entity2\"],\n"
        "\"Predicates\": [\"predicate1\", \"predicate2\"]\n"
        "}\n"
        "If nothing is found, use empty lists."
    )


def generate_with_hf(question: str, model_name: str = "meta-llama/Llama-3.1-8B-Instruct", max_new_tokens: int = 256, temperature: float = 0.0, retry_on_oom: bool = True) -> str:
    """Generate JSON extraction using a Hugging Face chat/instruction model via messages API when available."""
    instruction = build_instruction()
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": question},
    ]

    while True:
        try:
            txt_gen = pipeline(
                task="text-generation",
                model=model_name,
                tokenizer=model_name,
                device_map="auto" if torch.cuda.is_available() else None,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
            )

            # Prefer chat-style messages call when supported
            try:
                out = txt_gen(
                    messages,
                    max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0.0,
                    temperature=temperature,
                )
                first = out[0] if isinstance(out, list) and out else out
                text = first.get("generated_text", first)
                # Some pipelines return a list of turns; try to get assistant content
                if isinstance(text, list):
                    # Find last assistant message
                    assistant_msgs = [m for m in text if isinstance(m, dict) and m.get("role") == "assistant"]
                    if assistant_msgs:
                        return str(assistant_msgs[-1].get("content", "")).strip()
                    return str(text[-1]).strip()
                return str(text).strip()
            except TypeError:
                # Fallback: build a single-string prompt if messages-format isn't supported
                prompt = (
                    f"System: {instruction}\n\n"
                    f"User: {question}\n\n"
                    f"Assistant:"
                )
                out = txt_gen(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0.0,
                    temperature=temperature,
                    return_full_text=False,
                )
                return out[0]["generated_text"].strip()

        except RuntimeError as e:
            # Simple OOM backoff helper
            if retry_on_oom and ("CUDA out of memory" in str(e) or "out of memory" in str(e)):
                print("Model OOM encountered. Waiting 30s before retry with CPU-only...")
                time.sleep(30)
                retry_on_oom = False
                model = AutoModelForCausalLM.from_pretrained(model_name)
                tok = AutoTokenizer.from_pretrained(model_name)
                txt_gen = pipeline(task="text-generation", model=model, tokenizer=tok, device=-1)
                out = txt_gen(messages, max_new_tokens=max_new_tokens, do_sample=False)
                first = out[0] if isinstance(out, list) and out else out
                text = first.get("generated_text", first)
                return str(text).strip()
            raise


def validate_response(text: str) -> Tuple[List[str], List[str]]:
    """Try to extract and validate JSON with the expected schema."""
    s = text.strip()
    # If the model adds prose, try to locate the JSON object
    if not (s.startswith("{") and s.endswith("}")):
        try:
            start = s.index("{")
            end = s.rindex("}") + 1
            s = s[start:end]
        except ValueError:
            return [], []

    try:
        data = json.loads(s)
        ExtractionResult(**data)
        entities = [str(x).strip() for x in data.get("Entities", []) if str(x).strip()]
        predicates = [str(x).strip() for x in data.get("Predicates", []) if str(x).strip()]
        return entities, predicates
    except (json.JSONDecodeError, ValidationError):
        return [], []


def main(question: str, model_name: str = "meta-llama/Llama-3.1-8B-Instruct"):
    raw = generate_with_hf(question, model_name=model_name)
    print(f"Raw HF output:\n{raw}\n")
    entities, predicates = validate_response(raw)

    candidate_entity_ls, candidate_entity_embeddings = load_or_compute_embeddings(CANDIDATE_ENTITIES_PATH, ENTITY_EMBEDDINGS_PATH)
    candidate_property_ls, candidate_property_embeddings = load_or_compute_embeddings(CANDIDATE_PROPERTIES_PATH, PROPERTY_EMBEDDINGS_PATH)

    entity_matches = get_top_k_candidates(entities, candidate_entity_ls, candidate_entity_embeddings)
    property_matches = get_top_k_candidates(predicates, candidate_property_ls, candidate_property_embeddings)

    print("\n----- Matching Results via Vector Similarity -----")
    print("\nEntity Matches for:", question)
    print(entity_matches)
    print("\nPredicate Matches for:", question)
    print(property_matches)
    return entity_matches, property_matches


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task1 (HF): Entity/Property extraction via Hugging Face + matching")
    parser.add_argument("--question", required=True, help="Input question text")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct", help="HF model name (chat/instruct)")
    args = parser.parse_args()

    main(args.question, model_name=args.model)


