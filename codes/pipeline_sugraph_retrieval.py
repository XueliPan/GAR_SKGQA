import argparse
import sys
import os
import json
import pandas as pd
from rdflib import Graph, Namespace

# Local imports
from task1_entity_propterty_matching import main as task1_main
from task1_huggingface import main as task1_main_huggingface
from task2_subgraph_extraction import extract_n_hop_rdflib


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


def run_pipeline(question: str,
                 entity_label_csv: str,
                 orkg_graph_path: str,
                 hops: int = 1,
                 direction: str = "both",
                 include_literals: bool = True,
                 topk_candidates: int = 1,
                 topk_entities: int = 1,
                 output_ttl: str = "subgraph.ttl") -> dict:
    """
    1) Run task1 to get candidate entities/properties
    2) Map top entity labels to IRIs
    3) Extract n-hop subgraph(s) around each IRI and merge
    4) Save merged TTL
    Returns a summary dict with selected IRIs and output path.
    """
    # Step 1: Task1
    entity_matches, property_matches = task1_main(question)

    # Extract top-k candidate entity labels per source from task1 output DataFrame
    candidate_labels = []
    if entity_matches is not None and not entity_matches.empty:
        grouped = entity_matches.groupby("source_entity_property", sort=False)
        for _, group in grouped:
            top_rows = group.head(max(1, int(topk_candidates)))
            for _, row in top_rows.iterrows():
                candidate_labels.append(row["candidate_entity_property"])  # keep order

    # Step 2: map labels -> IRIs
    label_df = load_label_to_iri_map(entity_label_csv)
    seed_iris = labels_to_iris(candidate_labels, label_df, max_entities=topk_entities)

    print(f"Candidate labels from task1: {candidate_labels}")
    print(f"Mapped seed IRIs: {seed_iris}")

    # Step 3: subgraph extraction
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

    # Step 4: save
    merged.serialize(destination=output_ttl, format="turtle")

    return {
        "question": question,
        "seed_iris": seed_iris,
        "output_ttl": output_ttl,
        "triples": len(merged)
    }


def run_batch(input_csv: str,
              entity_label_csv: str,
              orkg_graph_path: str,
              hops: int,
              direction: str,
              include_literals: bool,
              topk_candidates: int,
              topk_entities: int,
              output_dir: str,
              summary_csv: str) -> None:
    """
    Process a CSV with columns: question_id, question_string, sparql_query.
    For each row, run the pipeline and write a per-question TTL under output_dir.
    Save a summary CSV with results and metadata.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_csv)
    required_cols = {"question_id", "question_string", "sparql_query"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {input_csv}: {sorted(list(missing))}")

    summaries = []
    for _, row in df.iterrows():
        qid = row["question_id"]
        question = row["question_string"]
        out_path = os.path.join(output_dir, f"{qid}_subgraph.ttl")
        summary = run_pipeline(
            question=question,
            entity_label_csv=entity_label_csv,
            orkg_graph_path=orkg_graph_path,
            hops=hops,
            direction=direction,
            include_literals=include_literals,
            topk_candidates=topk_candidates,
            topk_entities=topk_entities,
            output_ttl=out_path,
        )
        summary.update({
            "question_id": qid,
            "sparql_query": row.get("sparql_query", "")
        })
        summaries.append(summary)

    pd.DataFrame(summaries)[[
        "question_id",
        "question",
        "sparql_query",
        "seed_iris",
        "output_ttl",
        "triples"
    ]].to_csv(summary_csv, index=False)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pipeline: Task1 (entity/property candidates) -> Task2 (n-hop subgraph)")
    parser.add_argument("--input_csv", required=True, help="CSV with columns: question_id,question_string,sparql_query")
    parser.add_argument("--entity_label_csv", default="/Users/sherrypan/GitHub/GAR_SKGQA/datasets/sciqa/project_data/orkg_entity_labels.csv", help="CSV with iri,label rows")
    parser.add_argument("--orkg_rdf_dump", default="/Users/sherrypan/GitHub/GAR_SKGQA/datasets/sciqa/project_data/orkg-dump-14-02-2023.nt", help="Path to ORKG N-Triples dump (.nt)")
    parser.add_argument("--hops", type=int, default=1, help="Number of hops for subgraph extraction")
    parser.add_argument("--direction", choices=["in", "out", "both"], default="both", help="Traversal direction")
    parser.add_argument("--include_literals", type=str.lower, choices=["true", "false"], default="true", help="Include literal objects in subgraph (true/false)")
    parser.add_argument("--topk_candidates", type=int, default=1, help="Top-k candidates per source to take from task1 results")
    parser.add_argument("--topk_entities", type=int, default=1, help="Top entities to use as seeds")
    parser.add_argument("--output_dir", default="/Users/sherrypan/GitHub/GAR_SKGQA/results/subgraphs/hop1_entity_only", help="Directory for per-question TTL outputs")
    parser.add_argument("--summary_csv", default="/Users/sherrypan/GitHub/GAR_SKGQA/results/subgraphs/subgraph_summary_hop1_entity_only.csv", help="Path to write pipeline summary CSV")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    include_literals = True if args.include_literals == "true" else False

    run_batch(
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
    )
    print(json.dumps({"status": "ok", "summary_csv": args.summary_csv, "output_dir": args.output_dir}, indent=2))


if __name__ == "__main__":
    main()


