"""
DBLP SPARQL generation pipeline for subgraph-based questions (without execution).

Input CSV (output of DBLP subgraph extraction) is expected to have columns:
    id, question, construct_query, ground_truth_sparql, entities, relations,
    triples, subgraph_ttl, status, error_message

For each row we:
  1) Build an LLM prompt using the natural-language question and the subgraph
     (in Turtle) that was previously extracted.
  2) Generate a SPARQL (SELECT or ASK) query that can answer the question.
  3) Append the generated query as a new column and incrementally persist the CSV.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

import pandas as pd
from dotenv import load_dotenv

from helper import dblp_prefixes

try:  # optional runtime dependencies (LLM backends)
    import torch  # type: ignore
except ImportError:  # pragma: no cover
    torch = None  # type: ignore

try:
    from transformers import pipeline as hf_pipeline  # type: ignore
except ImportError:  # pragma: no cover
    hf_pipeline = None  # type: ignore

try:
    from google import genai  # type: ignore
    from google.genai import types as genai_types  # type: ignore
except ImportError:  # pragma: no cover
    genai = None  # type: ignore
    genai_types = None  # type: ignore

try:
    from SPARQLWrapper import POST, TURTLE, SPARQLWrapper, JSON
except ImportError:  # pragma: no cover
    SPARQLWrapper = None  # type: ignore


load_dotenv()

# Soft checks for keys (same style as other scripts)
if not os.getenv("GEMINI_API_KEY"):
    print("Warning: GEMINI_API_KEY not found in environment variables.")
if not os.getenv("OPENAI_API_KEY"):
    print("Warning: OPENAI_API_KEY not found in environment variables.")


@dataclass
class DblpQuestionRecord:
    """
    One row from the subgraph summary CSV.
    """

    id: str
    question: str
    construct_query: Optional[str]
    ground_truth_sparql: Optional[str]
    entities: Optional[str]
    relations: Optional[str]
    triples: Optional[int]
    subgraph_ttl: Optional[str]
    status: Optional[str]
    error_message: Optional[str]


def _strip_markdown_fences(text: str) -> str:
    """
    Remove leading/trailing markdown code fences like ```sparql ... ``` or ``` ... ```.
    """
    if not text:
        return text
    s = text.strip()
    if s.startswith("```sparql"):
        s = s[len("```sparql") :].lstrip("\n\r ")
    elif s.startswith("```"):
        s = s[len("```") :].lstrip("\n\r ")
    if s.endswith("```"):
        s = s[: -len("```")].rstrip()
    return s

def get_property_schema(relations: List[str], schema_file: Path) -> str:
    """
    Given a list of relation IRIs, extract their property schema from the property schema file.
    Returns a string representation suitable for inclusion in the prompt.
    """
    if not schema_file.exists():
        return "N/A"
    df = pd.read_csv(schema_file)
    # given a list of properties, output their domain and range in the format <domain, property, range>
    output_list = []
    for prop in relations:
        prop_name = "dblp:" + prop.split("#")[-1].strip(">")
        row = df[df['property'] == prop_name]
        if not row.empty:
            domain = row['domain'].values[0]
            range_ = row['range'].values[0]
            output_list.append(f"<{domain}, {prop_name}, {range_}>")
    return "\n".join(output_list)

def get_entity_types(entities: str, endpoint_url: str) -> str:
    """
    Given a list of entity IRIs, query their types from the SPARQL endpoint.
    Returns a string representation suitable for inclusion in the prompt, in the format: [entityIRI: type1, type1]
    Local endpoint url example: "http://localhost:7021/sparql"
    dblp endpoint url example: "https://dblp.org/sparql"
    """
    if SPARQLWrapper is None:
        raise ImportError("SPARQLWrapper is required for querying entity types.")

    SPARQL_TEMPLATE = """
    SELECT ?type
    WHERE {{
    {entity_uri} <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> ?type .
    }}
    """
    entity_types_list = []
    for entity_uri in entities:
        query = SPARQL_TEMPLATE.format(entity_uri=entity_uri)
        wrapper = SPARQLWrapper(endpoint_url)
        wrapper.setMethod(POST)
        wrapper.setReturnFormat(JSON)
        wrapper.setQuery(query)

        results = wrapper.query().convert()
        types = [result["type"]["value"] for result in results["results"]["bindings"]]
        # Format the output
        # Abbreviate the type URIs to use the 'dblp:' prefix where applicable
        abbreviated_types = []
        for t in types:
            if t.startswith("https://dblp.org/rdf/schema#"):
                abbreviated_types.append("dblp:" + t.split("#")[-1])
            else:
                abbreviated_types.append(t)
        types = abbreviated_types
        entity_types = f"{entity_uri}: " + ", ".join(types)
        entity_types_list.append(entity_types)
    final_output = " \n".join(entity_types_list)
    return final_output

def load_subgraph_csv(path: Path) -> pd.DataFrame:
    """
    Load the CSV produced by the DBLP subgraph extraction pipeline.
    """
    df = pd.read_csv(path)
    required_cols = {
        "id",
        "question",
        "construct_query",
        "ground_truth_sparql",
        "entities",
        "relations",
        "triples",
        "subgraph_ttl",
        "status",
        "error_message",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in input CSV {path}: {sorted(missing)}")
    return df


def iter_records(df: pd.DataFrame):
    """
    Yield DblpQuestionRecord rows.
    """
    for _, row in df.iterrows():
        yield DblpQuestionRecord(
            id=str(row["id"]),
            question=str(row["question"]),
            construct_query=row.get("construct_query"),
            ground_truth_sparql=row.get("ground_truth_sparql"),
            entities=row.get("entities"),
            relations=row.get("relations"),
            triples=row.get("triples"),
            subgraph_ttl=row.get("subgraph_ttl"),
            status=row.get("status"),
            error_message=row.get("error_message"),
        )


def _parse_csv_list_field(value: Optional[str]) -> List[str]:
    """
    Parse a CSV field that stores a comma-separated list of IRIs into a Python list.
    Empty / NaN / None become [].
    """
    if not isinstance(value, str):
        return []
    parts = [p.strip() for p in value.split(",")]
    return [p for p in parts if p]


def build_generation_prompt(
    question: str,
    subgraph_ttl: str,
    property_schema: str,
    entities_type: str,
) -> str:
    """
    Build a DBLP-specific prompt for generating an answer SPARQL query.

    We reuse the idea from the SciQA task3 prompt: give the question + subgraph
    and ask the model to emit a single SPARQL (SELECT or ASK) query.
    """
    prefixes = dblp_prefixes().strip()

    prompt = f"""
### Role
You are an expert SPARQL generator for the DBLP knowledge graph.

### Task
Given a natural-language question, a schema describing the properties that can be used, and a set of entities and their types, and a local subgraph (in Turtle) that is
relevant to this question, generate a single SPARQL query that can be executed against the DBLP SPARQL endpoint to retrieve the answer.

### DBLP Prefixes
{prefixes}

### Natural-Language Question
{question}

### Schema 
{property_schema}

### Entities and Their Types 
{entities_type}

### Subgraph (Turtle)
{subgraph_ttl}

Output only the SPARQL query (no backticks, no markdown, no commentary, no prefixes).
    """.strip()
    return prompt


def generate_sparql(
    backend: str,
    model: str,
    prompt: str,
    temperature: float = 0.1,
) -> str:
    """
    Generic SPARQL generation using the same LLM backends as the DBLP subgraph code.
    """
    backend = backend.lower()

    if backend == "google":
        if genai is None or genai_types is None:
            raise ImportError("google-genai is not installed. Please `pip install google-genai`.")
        client = genai.Client()
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=genai_types.GenerateContentConfig(temperature=float(temperature)),
        )
        print(f"Full generation response: {response}")
        return response.text.strip()

    if backend == "huggingface":
        if hf_pipeline is None:
            raise ImportError("transformers is not installed. Please `pip install transformers`.")
        if torch is None:
            raise ImportError("torch is required for Hugging Face inference.")
        generator = hf_pipeline(
            "text-generation",
            model=model,
            dtype=torch.bfloat16 if hasattr(torch, "bfloat16") else None,
            trust_remote_code=True,
            device_map="auto",
        )
        # Use return_full_text=False to get only new tokens, not the prompt

        messages = [
            {"role": "user", "content": prompt}]
        outputs = generator(
            messages,
            temperature=max(float(temperature), 0.1), 
            max_new_tokens=512,
        )
        
        return outputs[0]["generated_text"][1]['content']

    raise ValueError(f"Unsupported backend '{backend}'. Use 'google' or 'huggingface'.")


def run_pipeline(
    input_csv: Path,
    output_csv: Path,
    backend: str,
    model: str,
    temperature: float,
    endpoint_url: str,
    property_schema_file: Path,
    limit: Optional[int] = None,
) -> None:
    """
    Main per-question pipeline: SPARQL generation only (no execution) with incremental save.
    """
    df = load_subgraph_csv(input_csv)
    if limit is not None:
        df = df.head(limit)

    records = list(iter_records(df))
    print(f"Loaded {len(records)} questions from {input_csv}")

    # Prepare output DataFrame by copying existing columns so we preserve them
    out_rows = []

    for idx, rec in enumerate(records, start=1):
        print(f"\n=== Processing question {rec.id} (#{idx}) ===")
        generated_sparql: Optional[str] = None
        generation_error: Optional[str] = None

        try:
            # Prepare schema and entity-type context for the prompt
            relations_list = _parse_csv_list_field(rec.relations)
            entities_list = _parse_csv_list_field(rec.entities)
            property_schema = get_property_schema(relations_list, property_schema_file)
            entities_type = get_entity_types(entities_list, endpoint_url)

            prompt = build_generation_prompt(
                question=rec.question,
                subgraph_ttl=rec.subgraph_ttl,
                property_schema=property_schema,
                entities_type=entities_type,
            )
            print(f"Built generation prompt for {rec.id}.")

            # Generate SPARQL with internal retry loop (to handle transient LLM failures)
            while True:
                try:
                    generated_sparql = generate_sparql(
                        backend=backend,
                        model=model,
                        prompt=prompt,
                        temperature=temperature,
                    )
                    generated_sparql = _strip_markdown_fences(generated_sparql)
                    break
                except Exception as e:
                    print(
                        f"Error generating SPARQL for {rec.id}: {e}. "
                        "Retrying in 5 seconds..."
                    )
                    import time

                    time.sleep(5)

            print(f"Generated SPARQL for {rec.id}:\n{generated_sparql}\n")

        except Exception as e:
            # Capture any failure and continue
            generation_error = f"Pipeline error: {e}"
            print(
                f"Error while processing question {rec.id} (#{idx}): {e}. "
                "Skipping generation but keeping row."
            )

        # Build output row by copying original columns and appending new ones
        base_row = df.loc[df["id"] == rec.id].iloc[0].to_dict()
        base_row.update(
            {
                "generated_sparql": generated_sparql,
                "generation_error": generation_error,
            }
        )
        out_rows.append(base_row)

        # Incremental persistence so we don't lose progress on long runs
        pd.DataFrame(out_rows).to_csv(output_csv, index=False)
        print(f"Progress saved to {output_csv} after question {rec.id} (#{idx})")
        print("##############################################")

    print(f"\nPipeline complete. Final results written to {output_csv}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DBLP SPARQL generation (without execution) over subgraph-based questions"
    )
    parser.add_argument(
        "--input_csv",
        type=Path,
        required=True,
        help="Input CSV from DBLP subgraph extraction (with subgraph_ttl column).",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        required=True,
        help="Output CSV path with generated SPARQL appended.",
    )
    parser.add_argument(
        "--backend",
        choices=["google", "huggingface"],
        default="google",
        help="LLM backend to use for SPARQL generation.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name or identifier for the chosen backend.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Decoding temperature for generation.",
    )
    parser.add_argument(
        "--endpoint_url",
        type=str,
        default="http://localhost:7020/sparql",
        help="DBLP SPARQL endpoint URL (used for querying entity types).",
    )
    parser.add_argument(
        "--property_schema_file",
        type=Path,
        default=Path("/Users/sherrypan/GitHub/GAR_SKGQA/datasets/dblp/project_data/dblp_property_schema.csv"),
        help="Path to the DBLP property schema CSV file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionally limit to the first N questions.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_pipeline(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        backend=args.backend,
        model=args.model,
        temperature=args.temperature,
        endpoint_url=args.endpoint_url,
        property_schema_file=args.property_schema_file,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()

