"""
DBLP subgraph extraction pipeline driven by LLM-generated CONSTRUCT queries.

Steps:
1. Read the DBLP question JSON into a normalized pandas DataFrame.
2. Prompt an LLM to generate a CONSTRUCT SPARQL query per question using
   the question text plus the provided entities/relations.
3. Execute each CONSTRUCT query against an RDF dump with rdflib or via a SPARQL endpoint.
4. Persist resulting subgraphs to Turtle files and log a summary CSV.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional
from urllib.error import HTTPError
from rdflib import Graph, URIRef, RDFS, Namespace
import requests
from typing import List, Dict, Union

from dotenv import load_dotenv

import pandas as pd

try:  # optional runtime dependencies
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore

try:
    from transformers import pipeline as hf_pipeline
except ImportError:  # pragma: no cover
    hf_pipeline = None  # type: ignore

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:  # pragma: no cover
    genai = None  # type: ignore
    genai_types = None  # type: ignore

try:
    from SPARQLWrapper import POST, TURTLE, SPARQLWrapper, JSON
except ImportError:  # pragma: no cover
    SPARQLWrapper = None  # type: ignore

try:
    from rdflib import Graph
except ImportError:  # pragma: no cover
    Graph = None  # type: ignore

load_dotenv()
# Check if API key is available
if not os.getenv('GEMINI_API_KEY'):
    print("Warning: GEMINI_API_KEY not found in environment variables.")
if not os.getenv('OPENAI_API_KEY'):
    print("Warning: OPENAI_API_KEY not found in environment variables.")



PROMPT_TEMPLATE = """You are given:
1. A natural-language question.
2. A schema describing the ONLY allowed properties.
3. A set of entities and their types.

First, judge whether the question can be answered using a SPARQL SELECT query or an ASK query

If the question requires a SELECT query, your task is to generate a SINGLE SPARQL CONSTRUCT query that retrieves the
MINIMAL SUBGRAPH needed to answer the input question.

### Natural-Language Question
{question}

### Schema 
{property_schema}

### Entities and Their Types 
{entities_type}

### Requirements
- You MUST use ONLY the properties listed in the schema.
- You MUST include ALL listed properties at least once in the query.
- You MUST include ALL provided entities, placed according to their types.
- The SPARQL query MUST retrieve the MINIMAL SUBGRAPH necessary to answer the input question.
- The query MUST be a syntactically valid CONSTRUCT query.
- **Important:** For triple patterns where the object type is a literal (`xsd:string`, `xsd:anyURI`, `xsd:gMonth`, `xsd:integer`, `xsd:gYear`):
  - The **subject must be a variable** (e.g., `?paper`, `?venue`).
  - **Do NOT include the literal value** in the CONSTRUCT or WHERE clauses.
  - **Do NOT use FILTER or any other literal comparison** in any form.  
    - Reason: The literal in the KG may differ from the question (e.g., abbreviations, formatting).
- **Do NOT invent new properties, classes, or relationships** beyond the schema.
- The generated query should only use variables for literals from the question; never attempt to match literals directly.
- Output ONLY the SPARQL query (no explanations, no comments, no FILTERs).
- Return strictly the SPARQL query, without commentary. 
- Remove any markdown formatting such as ```sparql and ```, and ensure the query is syntactically correct.

If the question requires a ASK query, ingore the provided properties.
Your task is to generate a SINGLE SPARQL CONSTRUCT query that retrieves the MINIMAL SUBGRAPH relevant to the most important entity for the question.
For example, if the question is asking about whether an author has orcid or whether a paper has two authors, the most important entity is the author or the paper respectively. And the CONSTRUCT query should retrieve all triples related to that entity, ignoring the provided properties.
"""

# read the ontology file for inclusion in the prompt
ONTOLOGY_PATH = Path("/Users/sherrypan/GitHub/GAR_SKGQA/datasets/dblp/project_data/dblp_schema.ttl")
if ONTOLOGY_PATH.exists():
    ONTOLOGY_CONTENT = ONTOLOGY_PATH.read_text(encoding="utf-8")
    PROMPT_TEMPLATE = PROMPT_TEMPLATE.replace("{ontology}", ONTOLOGY_CONTENT)
else:
    PROMPT_TEMPLATE = PROMPT_TEMPLATE.replace("{ontology}", "N/A")


DBLP_PREFIXES = """
PREFIX dblp: <https://dblp.org/rdf/schema#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX cito: <http://purl.org/spar/cito/>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX schema: <https://schema.org/>
"""


@dataclass
class QuestionRecord:
    id: str
    question_string: str
    paraphrased_question: Optional[str]
    sparql_query: Optional[str]
    template_id: Optional[str]
    entities: List[str]
    relations: List[str]
    temporal: Optional[bool]
    help_out: Optional[bool]


def load_questions_dataframe(json_path: Path) -> pd.DataFrame:
    """
    Transform the DBLP JSON file into a DataFrame with the requested columns.
    """
    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    rows = []
    for item in payload.get("questions", []):
        rows.append(
            {
                "id": item.get("id"),
                "question_string": item.get("question", {}).get("string"),
                "paraphrased_question": item.get("paraphrased_question", {}).get("string"),
                "sparql_query": item.get("query", {}).get("sparql"),
                "template_id": item.get("template_id"),
                "entities": item.get("entities", []),
                "relations": item.get("relations", []),
                "temporal": item.get("temporal"),
                "help_out": item.get("held_out"),
            }
        )

    df = pd.DataFrame(rows)
    required_cols = [
        "id",
        "question_string",
        "paraphrased_question",
        "sparql_query",
        "template_id",
        "entities",
        "relations",
        "temporal",
        "help_out",
    ]
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns after JSON transformation: {sorted(missing)}")
    return df[required_cols]


def question_records(df: pd.DataFrame) -> Iterable[QuestionRecord]:
    for _, row in df.iterrows():
        yield QuestionRecord(
            id=row["id"],
            question_string=row["question_string"],
            paraphrased_question=row.get("paraphrased_question"),
            sparql_query=row.get("sparql_query"),
            template_id=row.get("template_id"),
            entities=row.get("entities") or [],
            relations=row.get("relations") or [],
            temporal=row.get("temporal"),
            help_out=row.get("help_out"),
        )


def get_zero_triple_list(file: Path) -> List[str]:
    """
    Given a file path, read the column with zero triple from the file and returned a list of question Id that have zero triples.
    """
    if not file.exists():
        return []
    zero_triple_list = []
    df = pd.read_csv(file)
    for _, row in df.iterrows():
        if row['triples'] == 0:
            zero_triple_list.append(row['id'])
    return zero_triple_list

def _escape_braces(value: Union[str, None]) -> str:
    """
    Escape curly braces in a string to make it safe for str.format().
    This prevents errors like "Single '}' encountered in format string".
    """
    if value is None:
        return ""
    return str(value).replace("{", "{{").replace("}", "}}")


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

def generate_construct_query(
    backend: str,
    model: str,
    prompt: str,
    temperature: float = 0.1,
) -> str:
    """
    Generate a SPARQL CONSTRUCT query using the specified LLM backend.
    """
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
            generated = generated[len(prompt):].strip()
        return generated

    raise ValueError(f"Unsupported backend '{backend}'. Use 'google' or 'huggingface'.")


def load_graph(rdf_path: Path) -> Graph:
    if Graph is None:
        raise ImportError("rdflib is required for local graph execution.")
    graph = Graph()
    if rdf_path.suffix == ".gz":
        with gzip.open(rdf_path, "rt", encoding="utf-8", errors="ignore") as handle:
            graph.parse(file=handle, format="nt")
    else:
        graph.parse(str(rdf_path))
    return graph


def execute_construct(
    query: str,
    graph: Optional[Graph] = None,
    endpoint_url: Optional[str] = None,
) -> Graph:
    # prefixes = """
    # PREFIX dblp: <https://dblp.org/rdf/schema#>
    # PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    # PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    # PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    # PREFIX wd: <http://www.wikidata.org/entity/>
    # PREFIX cito: <http://purl.org/spar/cito/>
    # PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
    # PREFIX schema: <https://schema.org/>
    # """
    # query = prefixes + query
    if endpoint_url:
        if SPARQLWrapper is None:
            raise ImportError("SPARQLWrapper is required for endpoint execution.")

        # Basic retry loop to handle HTTP 429 (rate limiting) from the endpoint.
        max_retries = 5
        base_wait_seconds = 10

        last_error: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                wrapper = SPARQLWrapper(endpoint_url)
                wrapper.setMethod(POST)
                wrapper.setReturnFormat(TURTLE)
                wrapper.setQuery(DBLP_PREFIXES + query)
                ttl = wrapper.query().convert()
                ttl_str = ttl.decode("utf-8") if isinstance(ttl, bytes) else ttl
                result_graph = Graph()
                result_graph.parse(data=ttl_str, format="turtle")
                return result_graph
            except HTTPError as e:
                last_error = e
                # 429 = Too Many Requests (rate limit)
                if e.code == 429 and attempt < max_retries:
                    wait_time = base_wait_seconds * attempt
                    print(
                        f"HTTP 429 from SPARQL endpoint (attempt {attempt}/{max_retries}). "
                        f"Sleeping {wait_time} seconds before retrying..."
                    )
                    time.sleep(wait_time)
                    continue
                # For non-429 or final attempt, re-raise
                raise
            except Exception as e:
                # For other transient network issues, you might also want to retry,
                # but for now just fail fast to surface the error.
                last_error = e
                raise

        # If we ever exit the loop without returning or raising, raise the last error.
        if last_error is not None:
            raise last_error

    if graph is None:
        raise ValueError("Provide either an rdflib Graph or a SPARQL endpoint URL.")

    result = graph.query(query)
    if hasattr(result, "graph") and isinstance(result.graph, Graph):
        return result.graph

    constructed = Graph()
    for triple in result:
        constructed.add(triple)
    return constructed


def save_graph(graph: Graph, output_path: Path) -> str:
    """
    Serialize the graph to Turtle, write it to disk, and return the TTL string.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ttl_data = graph.serialize(format="turtle")
    output_path.write_text(ttl_data, encoding="utf-8")
    return ttl_data


def run_pipeline(
    json_path: Path,
    rdf_path: Path,
    summary_csv: Path,
    zero_triple_file: Path,
    backend: str,
    model: str,
    temperature: float,
    property_schema_file: Path = Path("/Users/sherrypan/GitHub/GAR_SKGQA/datasets/dblp/project_data/dblp_property_schema.csv"),
    endpoint_url: Optional[str] = None,
    limit: Optional[int] = None,
) -> None:
    df = load_questions_dataframe(json_path)
    if limit is not None:
        df = df.head(limit)
    print(f"Loaded {len(df)} questions from {json_path}")
    graph = None if endpoint_url else load_graph(rdf_path)
    print(f"Loaded RDF graph from {rdf_path}")
    summaries = []
    zero_triple_list = get_zero_triple_list(zero_triple_file)
    print(f"zero_triple_list length: {len(zero_triple_list)}")

    for idx, record in enumerate(question_records(df[0:]), start=1):  # to resume from 0
        # if record.id in zero_triple_list then run the pipeline, otherwiese skip the questions.
        if record.id not in zero_triple_list:
            # print(f"Skipping question {record.id} (#{idx}) as it is not in the zero triple list.")
            continue
        try:
            prompt = PROMPT_TEMPLATE.format(
                question=_escape_braces(record.question_string),
                property_schema=_escape_braces(get_property_schema(record.relations, property_schema_file)),
                entities_type=_escape_braces(get_entity_types(record.entities, endpoint_url))
            )
            print(
                f"\nGenerated prompt for {record.id} (#{idx}):\n"
                # f"PROMPT start ******************\n{prompt}\nPROMPT end ******************"
            )
            if record.template_id:
                prompt += f"\nKnown template: {record.template_id}"

            # Generate CONSTRUCT query (with internal retry logic on LLM failures)
            while True:
                try:
                    construct_query = generate_construct_query(
                        backend=backend,
                        model=model,
                        prompt=prompt,
                        temperature=temperature,
                    )
                    break
                except Exception as e:
                    print(
                        f"Error generating CONSTRUCT query for {record.id}: {e}. "
                        "Retrying in 5 seconds..."
                    )
                    time.sleep(5)

            print(f"\nGenerated CONSTRUCT query for {record.id}:\n{construct_query}\n")

            # Sleep between requests to avoid rate limiting on LLM / endpoint
            time.sleep(10)

            # Execute CONSTRUCT query and serialize subgraph
            subgraph = execute_construct(
                query=construct_query, graph=graph, endpoint_url=endpoint_url
            )
            ttl_data = subgraph.serialize(format="turtle")

            summaries.append(
                {
                    "id": record.id,
                    "question": record.question_string,
                    "construct_query": construct_query,
                    "ground_truth_sparql": record.sparql_query,
                    "entities": ", ".join(record.entities) if record.entities else "",
                    "relations": ", ".join(record.relations) if record.relations else "",
                    "triples": len(subgraph),
                    "subgraph_ttl": ttl_data,
                    "status": "ok",
                    "error_message": "",
                }
            )
            print(f"Processed subgraph for {record.id} ({len(subgraph)} triples)")

        except Exception as e:
            # Catch any failure for this record so the pipeline can continue.
            print(
                f"Error while processing question {record.id} (#{idx}): {e}. "
                "Skipping this question and continuing..."
            )
            summaries.append(
                {
                    "id": record.id,
                    "question": record.question_string,
                    "construct_query": None,
                    "ground_truth_sparql": record.sparql_query,
                    "entities": ", ".join(record.entities) if record.entities else "",
                    "relations": ", ".join(record.relations) if record.relations else "",
                    "triples": 0,
                    "subgraph_ttl": "",
                    "status": "error",
                    "error_message": str(e),
                }
            )

        # Incremental persistence so we don't lose progress on long runs.
        pd.DataFrame(summaries).to_csv(summary_csv, index=False)
        print(f"Progress saved to {summary_csv} after question {record.id} (#{idx})")

    print(f"Pipeline complete. Final summary written to {summary_csv}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LLM-driven DBLP subgraph extraction")
    parser.add_argument(
        "--json_path",
        type=Path,
        default=Path("/Users/sherrypan/GitHub/GAR_SKGQA/datasets/dblp/test/questions.json"),
    )
    parser.add_argument(
        "--rdf_path",
        type=Path,
        default=Path("/Users/sherrypan/GitHub/GAR_SKGQA/datasets/dblp/dblp-2022-03-01.nt.gz"),
    )
    parser.add_argument("--summary_csv", type=Path)
    parser.add_argument("--zero_triple_file", type=Path, default=Path("/Users/sherrypan/GitHub/GAR_SKGQA/results/dblp/zero_triple_question_247.csv"))  
    parser.add_argument(
        "--property_schema_file",
        type=Path,
        default=Path("/Users/sherrypan/GitHub/GAR_SKGQA/datasets/dblp/project_data/dblp_property_schema.csv"),
        help="Path to the property schema CSV file.",
    )
    parser.add_argument("--backend", choices=["google", "huggingface"], default="google")
    parser.add_argument("--model", type=str, required=True, help="LLM model name or identifier.")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--endpoint_url", type=str, default=None, help="Optional SPARQL endpoint URL.")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N questions.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_pipeline(
        json_path=args.json_path,
        rdf_path=args.rdf_path,
        summary_csv=args.summary_csv,
        zero_triple_file=args.zero_triple_file,
        backend=args.backend,
        model=args.model,
        temperature=args.temperature,
        property_schema_file=args.property_schema_file,
        endpoint_url=args.endpoint_url,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()