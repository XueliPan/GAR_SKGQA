import os
import argparse
from typing import Optional, Dict, Tuple
import pandas as pd
from google import genai
from dotenv import load_dotenv
from google.genai import types
from helper import orkg_prefixes


def _strip_markdown_fences(text: str) -> str:
    """
    Remove leading/trailing markdown code fences like ```sparql ... ``` or ``` ... ```.
    Preserves the inner query content.
    """
    if not text:
        return text
    s = text.strip()
    # Remove starting fence variants
    if s.startswith("```sparql"):
        s = s[len("```sparql"):].lstrip("\n\r ")
    elif s.startswith("```"):
        s = s[len("```"):].lstrip("\n\r ")
    # Remove trailing fence
    if s.endswith("```"):
        s = s[: -len("```")].rstrip()
    return s


def generate_orkg_sparql(
    target_question: str,
    subgraph_turtle: str,
    example_question: Optional[str] = None,
    example_sparql: Optional[str] = None,
    ontology_turtle: Optional[str] = None,
) -> str:
    """
    Generate a SPARQL query for ORKG using Gemini with a modular prompt that includes:
    - Required: input question, subgraph (Turtle), ORKG prefixes
    - Optional: example question and example query (one-shot), ontology (Turtle)

    Args:
        target_question: Natural language question to convert to SPARQL. (required)
        subgraph_turtle: Turtle text of the subgraph relevant to the question. (required)
        example_question: Optional example question for one-shot guidance.
        example_sparql: Optional SPARQL matching the example question.
        ontology_turtle: Optional ontology snippet in Turtle.

    Returns:
        The generated SPARQL query string.
    """

    # Required ORKG prefixes
    PREFIXES = orkg_prefixes().strip()

    # Assemble optional sections
    example_block = ""
    if example_question and example_sparql:
        example_block = f"""
### Example (Optional One-Shot)
Question: {example_question}
SPARQL:
{example_sparql.strip()}
""".strip()

    ontology_block = ""
    if ontology_turtle:
        ontology_block = f"""
### Ontology (Optional, Turtle)
{ontology_turtle.strip()}
""".strip()

    # Construct final prompt with required and optional sections
    FULL_PROMPT = f"""
### Role
You are an expert SPARQL generator for the ORKG (Open Research Knowledge Graph).

### Instructions
Given an input Question and a Subgraph in Turtle format, your task is to generate a SPARQL query that can be executed against the ORKG SPARQL endpoint to retrieve the answer. Use the provided prefixes and ensure the query is syntactically correct and semantically aligned with the question.
The output must:
- Don't include all necessary PREFIX declarations in the output, as they are already provided above.
- Be a single SPARQL query only, with no extra commentary
- Don't use any comments in the SPARQL

### Question (Required)
{target_question}

### ORKG Prefixes (Required)
{PREFIXES}

{subgraph_turtle.strip()}

{example_block}

{ontology_block}


""".strip()

    try:
        client = genai.Client()
        print(f"\n####################################")
        print(f"Full prompt:\n{FULL_PROMPT}")
        print(f"\n####################################")
        response = client.models.generate_content(
            model='gemini-2.5-pro',
            contents=FULL_PROMPT,
            config=types.GenerateContentConfig(temperature=0.1)
        )
        return response.text.strip()
    except Exception as e:
        return f"An error occurred (check API key and internet connection): {e}"


# --- EXAMPLE USAGE ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Task3: Generate SPARQL from question + subgraph")
    parser.add_argument("--input_csv", required=True, help="CSV with columns: question_id,question_string,sparql_query")
    parser.add_argument("--subgraph_dir", required=True, help="Directory containing per-question subgraphs named <question_id>_subgraph.ttl")
    parser.add_argument("--output_csv", required=True, help="Where to write sparql generation results CSV")
    parser.add_argument("--one_shot_csv", default=None, help="CSV with columns: id,test_question,test_query,best_train_question,best_train_query for per-question examples")
    parser.add_argument("--ontology_turtle_path", default=None, help="Optional path to ontology Turtle file")
    args = parser.parse_args()

    load_dotenv()
    if not os.getenv('GEMINI_API_KEY'):
        raise RuntimeError("Please set the 'GEMINI_API_KEY' environment variable.")

    ontology_turtle = None
    if args.ontology_turtle_path and os.path.exists(args.ontology_turtle_path):
        with open(args.ontology_turtle_path, 'r', encoding='utf-8') as f:
            ontology_turtle = f.read()

    df = pd.read_csv(args.input_csv)
    required_cols = {"question_id", "question_string", "sparql_query"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {args.input_csv}: {sorted(list(missing))}")

    # Load one-shot mapping if provided
    one_shot_map: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
    if args.one_shot_csv and os.path.exists(args.one_shot_csv):
        os_df = pd.read_csv(args.one_shot_csv)
        os_required = {"id", "test_question", "test_query", "best_train_question", "best_train_query"}
        os_missing = os_required - set(os_df.columns)
        if os_missing:
            raise ValueError(f"Missing required columns in {args.one_shot_csv}: {sorted(list(os_missing))}")
        # Build map from id (as string) to tuple(example_question, example_query)
        for _, r in os_df.iterrows():
            qid_key = str(r["id"]) if not pd.isna(r["id"]) else None
            if qid_key is None:
                continue
            one_shot_map[qid_key] = (
                None if pd.isna(r["best_train_question"]) else str(r["best_train_question"]),
                None if pd.isna(r["best_train_query"]) else str(r["best_train_query"]),
            )

    rows = []
    for _, row in df.iterrows():
        qid = str(row["question_id"])  # preserve naming
        question = row["question_string"]
        subgraph_path = os.path.join(args.subgraph_dir, f"{qid}_subgraph.ttl")
        if not os.path.exists(subgraph_path):
            gen_query = ""
            error = f"Missing subgraph file: {subgraph_path}"
            rows.append({
                "question_id": qid,
                "question_string": question,
                "gold_sparql": row.get("sparql_query", ""),
                "generated_sparql": gen_query,
                "error": error
            })
            continue

        with open(subgraph_path, 'r', encoding='utf-8') as f:
            subgraph_ttl = f.read()

        # Determine per-question example if available; otherwise none
        per_example_question = None
        per_example_sparql = None
        if qid in one_shot_map:
            ex_q, ex_s = one_shot_map[qid]
            per_example_question = ex_q
            per_example_sparql = ex_s

        gen_query = generate_orkg_sparql(
            target_question=question,
            subgraph_turtle=subgraph_ttl,
            example_question=per_example_question,
            example_sparql=per_example_sparql,
            ontology_turtle=ontology_turtle,
        )

        # Clean markdown fences if present
        gen_query = _strip_markdown_fences(gen_query)

        rows.append({
            "question_id": qid,
            "question_string": question,
            "gold_sparql": row.get("sparql_query", ""),
            "generated_sparql": gen_query,
            "error": ""
        })
        print(f"\nProcessed question_id={qid}")
        print(f"\nGenerated SPARQL:\n{gen_query}")
        # print(f"\nGold SPARQL:\n{row.get('sparql_query', '')}")
        print("-------------------------------------------------------------------------------")

    pd.DataFrame(rows)[[
        "question_id",
        "question_string",
        "gold_sparql",
        "generated_sparql",
        "error"
    ]].to_csv(args.output_csv, index=False)
    print(f"Wrote results to {args.output_csv}")