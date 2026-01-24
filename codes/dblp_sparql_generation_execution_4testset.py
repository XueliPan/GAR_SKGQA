"""
DBLP SPARQL generation + execution pipeline for subgraph-based questions.

Input CSV (output of DBLP subgraph extraction) is expected to have columns:
    id, question, construct_query, ground_truth_sparql, entities, relations,
    triples, subgraph_ttl, status, error_message

For each row we:
  1) Build an LLM prompt using the natural-language question and the subgraph
     (in Turtle) that was previously extracted.
  2) Generate a SPARQL (SELECT or ASK) query that can answer the question.
  3) Execute the generated SPARQL against a DBLP SPARQL endpoint.
  4) Append the generated query and execution result as new columns and
     incrementally persist the CSV.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

import pandas as pd
from dotenv import load_dotenv

from helper import dblp_prefixes
from subgraph_extraction_dblp_prompt2 import get_property_schema, get_entity_types

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
    from SPARQLWrapper import SPARQLWrapper, JSON  # type: ignore
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
        outputs = generator(prompt, max_new_tokens=512, do_sample=False, temperature=float(temperature))
        generated = outputs[0].get("generated_text", "").strip()
        if generated.startswith(prompt):
            generated = generated[len(prompt) :].strip()
        return generated

    raise ValueError(f"Unsupported backend '{backend}'. Use 'google' or 'huggingface'.")


def execute_sparql(
    sparql_query: str,
    endpoint_url: str,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Execute a SPARQL query against a DBLP endpoint and normalize the JSON result.

    Returns:
        (result_json_string | None, error_message | None)
    """
    if SPARQLWrapper is None:
        raise ImportError("SPARQLWrapper is required for SPARQL execution.")

    prefixes = dblp_prefixes()
    sparql_text = f"{prefixes}\n{sparql_query}"

    try:
        wrapper = SPARQLWrapper(endpoint_url)
        wrapper.setQuery(sparql_text)
        wrapper.setReturnFormat(JSON)
        results = wrapper.query().convert()

        # Handle ASK queries (boolean result)
        if isinstance(results, dict) and "boolean" in results:
            return json.dumps({"boolean": results["boolean"]}), None

        # Normalize binding-based results a bit (remove non-essential keys)
        if isinstance(results, dict):
            head = results.get("head")
            if isinstance(head, dict):
                head = dict(head)
                head.pop("link", None)
                results["head"] = head
            res_obj = results.get("results")
            if isinstance(res_obj, dict):
                res_obj = dict(res_obj)
                res_obj.pop("distinct", None)
                res_obj.pop("ordered", None)
                results["results"] = res_obj

        return json.dumps(results), None
    except Exception as e:  # pragma: no cover - network / endpoint dependent
        return None, str(e)


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
    Main per-question pipeline: SPARQL generation + execution with incremental save.
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
        exec_result: Optional[str] = None
        exec_error: Optional[str] = None

        # # If subgraph extraction previously failed, skip generation but keep the row
        # if not rec.subgraph_ttl or not isinstance(rec.subgraph_ttl, str) or not rec.subgraph_ttl.strip():
        #     print(f"Skipping SPARQL generation for {rec.id} because subgraph_ttl is empty.")
        #     out_rows.append(
        #         {
        #             **df.loc[df["id"] == rec.id].iloc[0].to_dict(),
        #             "generated_sparql": None,
        #             "execution_results": None,
        #             "execution_error": "No subgraph_ttl available",
        #         }
        #     )
        #     pd.DataFrame(out_rows).to_csv(output_csv, index=False)
        #     print(f"Progress saved to {output_csv} after question {rec.id} (#{idx})")
        #     continue

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
            print(f"Prompt:\n{prompt}\n")

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

            # Execute the generated SPARQL
            if generated_sparql and generated_sparql.strip():
                exec_result, exec_error = execute_sparql(
                    sparql_query=generated_sparql,
                    endpoint_url=endpoint_url,
                )
                print(f"Executed SPARQL for {rec.id}. Error: {exec_error}")

        except Exception as e:
            # Capture any failure and continue
            exec_error = f"Pipeline error: {e}"
            print(
                f"Error while processing question {rec.id} (#{idx}): {e}. "
                "Skipping execution but keeping generated data (if any)."
            )

        # Build output row by copying original columns and appending new ones
        base_row = df.loc[df["id"] == rec.id].iloc[0].to_dict()
        base_row.update(
            {
                "generated_sparql": generated_sparql,
                "execution_results": exec_result,
                "execution_error": exec_error,
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
        description="DBLP SPARQL generation + execution over subgraph-based questions"
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
        help="Output CSV path with generated SPARQL and execution results appended.",
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
        default="http://localhost:7020",
        help="LocalDBLP SPARQL endpoint URL.",
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


