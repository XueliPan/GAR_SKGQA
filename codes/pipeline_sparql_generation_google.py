import argparse
import sys
import os
import json
import pandas as pd
from rdflib import Graph, Namespace
from typing import Optional, Dict, Tuple, List
from dotenv import load_dotenv

# Local imports - reusing existing code without modifications
from task1_entity_propterty_matching import main as task1_main
from task2_subgraph_extraction import extract_n_hop_rdflib
from task3_sparql_generation import generate_orkg_sparql, _strip_markdown_fences


def load_label_to_iri_map(csv_path: str) -> pd.DataFrame:
    """
    Load a simple two-column CSV mapping of IRI,label without any normalization.
    Returns a DataFrame with columns: iri, label.
    """
    df = pd.read_csv(csv_path, header=None, names=["iri", "label"])
    df = df.dropna(subset=["iri", "label"]).copy()
    return df[["iri", "label"]]


def labels_to_iris(candidate_labels, label_df: pd.DataFrame, max_entities: int = 3) -> list:
    """
    Map candidate labels (strings) to IRIs by exact labels returned from task1, entity/property matching.
    Returns up to max_entities IRIs in the same order as input labels when matched.
    """
    iris = []
    if not candidate_labels:
        return iris
    for label in candidate_labels:
        # exact match against raw label, no normalization
        hit = label_df[label_df["label"] == label]
        if not hit.empty:
            iris.append(hit.iloc[0]["iri"])
            if len(iris) >= max_entities:
                break
    return iris


def create_enhanced_prompt_context(
    question: str,
    entity_matches: pd.DataFrame,
    property_matches: pd.DataFrame,
    seed_iris: List[str],
    subgraph_turtle: str
) -> str:
    """
    Create enhanced context for SPARQL generation including candidate entities/properties and their IRIs.
    """
    context_parts = []
    
    # Add candidate entities section
    if entity_matches is not None and not entity_matches.empty:
        context_parts.append("### Candidate Entities, their labels(rdfs:label) and similarity scores")
        entity_context = []
        for _, row in entity_matches.head(5).iterrows():  # Limit to top 5
            source = row["source_entity_property"]
            candidate = row["candidate_entity_property"]
            score = row["similarity_score"]
            entity_context.append(f"- Source: '{source}' → Candidate: '{candidate}' (similarity: {score})")
        context_parts.extend(entity_context)
        context_parts.append("")
    
    # Add candidate properties section
    if property_matches is not None and not property_matches.empty:
        context_parts.append("### Candidate Properties, their labels(rdfs:label) and similarity scores")
        property_context = []
        for _, row in property_matches.head(5).iterrows():  # Limit to top 5
            source = row["source_entity_property"]
            candidate = row["candidate_entity_property"]
            score = row["similarity_score"]
            property_context.append(f"- Source: '{source}' → Candidate: '{candidate}' (similarity: {score})")
        context_parts.extend(property_context)
        context_parts.append("")
    
    # Add seed IRIs section
    if seed_iris:
        context_parts.append("### Seed IRIs for Subgraph Extraction")
        for iri in seed_iris:
            context_parts.append(f"- {iri}")
        context_parts.append("")
    
    return "\n".join(context_parts)


def run_sparql_generation_pipeline(
    question: str,
    entity_label_csv: str,
    orkg_graph_path: str,
    hops: int = 1,
    direction: str = "both",
    include_literals: bool = True,
    topk_candidates: int = 1,
    topk_entities: int = 1,
    example_question: Optional[str] = None,
    example_sparql: Optional[str] = None,
    ontology_turtle: Optional[str] = None,
    output_ttl: str = "subgraph.ttl"
) -> dict:
    """
    Complete pipeline for SPARQL generation:
    1) Run task1 to get candidate entities/properties
    2) Map top entity labels to IRIs
    3) Extract n-hop subgraph(s) around each IRI and merge
    4) Generate SPARQL query using enhanced prompt with all context
    5) Save merged TTL and return results
    
    Returns a summary dict with selected IRIs, generated SPARQL, and output path.
    """
    print(f"\nProcessing question: {question}")
    
    # Step 1: Task1 - Entity and Property Extraction and Matching
    print("\nStep 1: Extracting entities and properties, finding candidates...")
    entity_matches, property_matches = task1_main(question)
    
    # Extract top-k candidate entity labels per source from task1 output DataFrame
    candidate_labels = []
    if entity_matches is not None and not entity_matches.empty:
        grouped = entity_matches.groupby("source_entity_property", sort=False)
        for _, group in grouped:
            top_rows = group.head(max(1, int(topk_candidates)))
            for _, row in top_rows.iterrows():
                candidate_labels.append(row["candidate_entity_property"])  # keep order

    # Step 2: Map labels -> IRIs
    print(f"\nStep 2: Mapping candidate labels to IRIs...")
    label_df = load_label_to_iri_map(entity_label_csv)
    seed_iris = labels_to_iris(candidate_labels, label_df, max_entities=topk_entities)

    print(f"Candidate labels from task1: {candidate_labels}")
    print(f"Mapped seed IRIs: {seed_iris}")

    # Step 3: Subgraph extraction
    print(f"\nStep 3: Extracting subgraph...")
    g = Graph()
    # Format inferred by extension: .nt for N-Triples
    g.parse(orkg_graph_path, format="nt")

    merged = Graph()
    for iri in seed_iris:
        print(f"Extracting {hops}-hop subgraph around seed IRI: {iri}")
        sub = extract_n_hop_rdflib(g, iri, n=hops, direction=direction, include_literals=include_literals)
        for t in sub:
            merged.add(t)

    # Add ORKGP prefix if present
    ORKGP = Namespace("http://orkg.org/orkg/predicate/")
    merged.bind("orkgp", ORKGP)

    # Step 4: Save subgraph
    print(f"\nStep 4: Saving subgraph")
    merged.serialize(destination=output_ttl, format="turtle")
    subgraph_turtle = open(output_ttl, 'r', encoding='utf-8').read()
    print(f"Saved subgraph to {output_ttl}")

    # Step 5: Generate SPARQL with enhanced context
    print(f"\nStep 5: Generating SPARQL query...")
    
    try:
        # Create enhanced context for the prompt
        enhanced_context = create_enhanced_prompt_context(
            question, entity_matches, property_matches, seed_iris, subgraph_turtle
        )
        
        # Combine enhanced context with subgraph for the prompt
        enhanced_subgraph = f"{enhanced_context}\n### Subgraph (in Turtle format)\n{subgraph_turtle}"
        
        print(f"Calling generate_orkg_sparql with enhanced context...")
        # Generate SPARQL using the existing function with enhanced subgraph
        generated_sparql = generate_orkg_sparql(
            target_question=question,
            subgraph_turtle=enhanced_subgraph,
            example_question=example_question,
            example_sparql=example_sparql,
            ontology_turtle=ontology_turtle,
        )
        
        # Clean markdown fences if present
        generated_sparql = _strip_markdown_fences(generated_sparql)
        print(f"Generated SPARQL: {generated_sparql}")
        
    except Exception as e:
        print(f"Error in Step 5 (SPARQL generation): {str(e)}")
        print(f"Exception type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        generated_sparql = f"Error generating SPARQL: {str(e)}"

    return {
        "question": question,
        "seed_iris": seed_iris,
        "subgraph_turtle": subgraph_turtle,
        "triples": len(merged),
        "generated_sparql": generated_sparql,
        "entity_matches": entity_matches.to_dict('records') if entity_matches is not None and not entity_matches.empty else [],
        "property_matches": property_matches.to_dict('records') if property_matches is not None and not property_matches.empty else []
    }


def run_batch_sparql_generation(
    input_csv: str,
    entity_label_csv: str,
    orkg_graph_path: str,
    hops: int,
    direction: str,
    include_literals: bool,
    topk_candidates: int,
    topk_entities: int,
    output_dir: str,
    summary_csv: str,
    one_shot_csv: Optional[str] = None,
    ontology_turtle_path: Optional[str] = None
) -> None:
    """
    Process a CSV with columns: question_id, question_string, sparql_query.
    For each row, run the complete SPARQL generation pipeline and write results.
    Save a summary CSV with results and metadata.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_csv)
    required_cols = {"question_id", "question_string", "sparql_query"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {input_csv}: {sorted(list(missing))}")

    # Load one-shot mapping if provided
    one_shot_map: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
    if one_shot_csv and os.path.exists(one_shot_csv):
        os_df = pd.read_csv(one_shot_csv)
        os_required = {"id", "test_question", "test_query", "best_train_question", "best_train_query"}
        os_missing = os_required - set(os_df.columns)
        if os_missing:
            raise ValueError(f"Missing required columns in {one_shot_csv}: {sorted(list(os_missing))}")
        # Build map from id (as string) to tuple(example_question, example_query)
        for _, r in os_df.iterrows():
            qid_key = str(r["id"]) if not pd.isna(r["id"]) else None
            if qid_key is None:
                continue
            one_shot_map[qid_key] = (
                None if pd.isna(r["best_train_question"]) else str(r["best_train_question"]),
                None if pd.isna(r["best_train_query"]) else str(r["best_train_query"]),
            )

    # Load ontology if provided
    ontology_turtle = None
    if ontology_turtle_path and os.path.exists(ontology_turtle_path):
        with open(ontology_turtle_path, 'r', encoding='utf-8') as f:
            ontology_turtle = f.read()

    summaries = []
    for idx, row in df.iterrows():
        print(f"Processing the {idx+1}th question")
        qid = row["question_id"]
        question = row["question_string"]
        out_path = os.path.join(output_dir, f"{qid}_subgraph.ttl")
        
        # Determine per-question example if available
        per_example_question = None
        per_example_sparql = None
        if str(qid) in one_shot_map:
            ex_q, ex_s = one_shot_map[str(qid)]
            per_example_question = ex_q
            per_example_sparql = ex_s

        try:
            summary = run_sparql_generation_pipeline(
                question=question,
                entity_label_csv=entity_label_csv,
                orkg_graph_path=orkg_graph_path,
                hops=hops,
                direction=direction,
                include_literals=include_literals,
                topk_candidates=topk_candidates,
                topk_entities=topk_entities,
                example_question=per_example_question,
                example_sparql=per_example_sparql,
                ontology_turtle=ontology_turtle,
                output_ttl=out_path,
            )
            summary.update({
                "question_id": qid,
                "gold_sparql": row.get("sparql_query", ""),
                "error": ""
            })
        except Exception as e:
            print(f"Exception occurred for question_id={qid}: {str(e)}")
            print(f"Exception type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            summary = {
                "question_id": qid,
                "question": question,
                "gold_sparql": row.get("sparql_query", ""),
                "generated_sparql": "",
                "seed_iris": [],
                "subgraph_turtle": "",
                "triples": 0,
                "entity_matches": [],
                "property_matches": [],
                "error": str(e)
            }
        
        summaries.append(summary)
        print(f"Completed processing question_id={qid}")
        print("-------------------------------------------------------------------------------")

    # Save results
    result_df = pd.DataFrame(summaries)
    result_df[[
        "question_id",
        "question",
        "gold_sparql",
        "generated_sparql",
        "seed_iris",
        "subgraph_turtle",
        "triples",
        "entity_matches",
        "property_matches",
        "error"
    ]].to_csv(summary_csv, index=False)
    
    # Save detailed results with entity/property matches
    detailed_csv = summary_csv.replace('.csv', '_detailed.csv')
    result_df.to_csv(detailed_csv, index=False)
    
    print(f"Wrote summary results to {summary_csv}")
    print(f"Wrote detailed results to {detailed_csv}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Complete Pipeline: Task1 (entity/property candidates) -> Task2 (n-hop subgraph) -> Task3 (SPARQL generation)")
    parser.add_argument("--input_csv", required=True, help="CSV with columns: question_id,question_string,sparql_query")
    parser.add_argument("--entity_label_csv", default="/Users/sherrypan/GitHub/GAR_SKGQA/datasets/sciqa/project_data/orkg_entity_labels.csv", help="CSV with iri,label rows")
    parser.add_argument("--orkg_rdf_dump", default="/Users/sherrypan/GitHub/GAR_SKGQA/datasets/sciqa/project_data/orkg-dump-14-02-2023.nt", help="Path to ORKG N-Triples dump (.nt)")
    parser.add_argument("--hops", type=int, default=1, help="Number of hops for subgraph extraction")
    parser.add_argument("--direction", choices=["in", "out", "both"], default="both", help="Traversal direction")
    parser.add_argument("--include_literals", type=str.lower, choices=["true", "false"], default="true", help="Include literal objects in subgraph (true/false)")
    parser.add_argument("--topk_candidates", type=int, default=1, help="Top-k candidates per source to take from task1 results")
    parser.add_argument("--topk_entities", type=int, default=1, help="Top entities to use as seeds")
    parser.add_argument("--output_dir", default="/Users/sherrypan/GitHub/GAR_SKGQA/results/sparql_generation", help="Directory for per-question TTL outputs")
    parser.add_argument("--summary_csv", default="/Users/sherrypan/GitHub/GAR_SKGQA/results/sparql_generation_summary.csv", help="Path to write pipeline summary CSV")
    parser.add_argument("--one_shot_csv", default=None, help="CSV with columns: id,test_question,test_query,best_train_question,best_train_query for per-question examples")
    parser.add_argument("--ontology_turtle_path", default=None, help="Optional path to ontology Turtle file")
    return parser


def main():
    load_dotenv()
    # Check if API key is available
    if not os.getenv('GEMINI_API_KEY'):
        print("Warning: GEMINI_API_KEY not found in environment variables.")
    if not os.getenv('OPENAI_API_KEY'):
        print("Warning: OPENAI_API_KEY not found in environment variables.")
    
    parser = build_arg_parser()
    args = parser.parse_args()

    include_literals = True if args.include_literals == "true" else False

    run_batch_sparql_generation(
        input_csv=args.input_csv,
        entity_label_csv=args.entity_label_csv,
        orkg_graph_path=args.orkg_rdf_dump,
        hops=args.hops,
        direction=args.direction,
        include_literals=include_literals,
        topk_candidates=args.topk_candidates,
        topk_entities=args.topk_entities,
        output_dir=args.output_dir,
        summary_csv=args.summary_csv,
        one_shot_csv=args.one_shot_csv,
        ontology_turtle_path=args.ontology_turtle_path,
    )
    print(json.dumps({"status": "ok", "summary_csv": args.summary_csv, "output_dir": args.output_dir}, indent=2))


if __name__ == "__main__":
    main()
