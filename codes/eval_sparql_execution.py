import json
from sys import argv
import pandas as pd
from SPARQLWrapper import SPARQLWrapper, JSON

SPARQLPATH = "http://localhost:8890/sparql"

def test(sparql_text):
    try:
        sparql = SPARQLWrapper(SPARQLPATH)
        sparql_txt = sparql_text
        sparql.setQuery(sparql_txt)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        # Handle ASK queries that return a boolean result
        print("RAW RESULTS: ", results)
        if isinstance(results, dict) and "boolean" in results:
            return json.dumps({"boolean": results["boolean"]}), None
        # Remove non-essential keys when result is bindings-based
        if isinstance(results, dict):
            if "head" in results and isinstance(results["head"], dict):
                results["head"].pop("link", None)
            if "results" in results and isinstance(results["results"], dict):
                results["results"].pop("distinct", None)
                results["results"].pop("ordered", None)
        return json.dumps(results), None
    except Exception as e:
        # get the error message
        print(e)
        return None, str(e)

def main(input_file, output_file):
    # Load data from CSV file with expected columns
    # expected: question_id,question_string,gold_sparql,generated_sparql,seed_iris,subgraph_turtle,triples,entity_matches,property_matches,error
    df = pd.read_csv(input_file)

    required_cols = [
        "question_id",
        "question_string",
        "gold_sparql",
        "generated_sparql",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input CSV: {missing}")

    prefixes = """
    PREFIX orkgp: <http://orkg.org/orkg/predicate/>
    PREFIX orkgc: <http://orkg.org/orkg/class/>
    PREFIX orkgr: <http://orkg.org/orkg/resource/>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    """

    execution_results = []
    execution_errors = []
    for idx, row in df.iterrows():
        query_text = row.get("generated_sparql")
        if pd.isna(query_text) or not isinstance(query_text, str) or not query_text.strip():
            execution_results.append(None)
            execution_errors.append(None)
            continue
        result, error_msg = test(sparql_text=f"{prefixes}\n{query_text}")
        print(f"Query for: {row.get('question_id')}")
        print(result)
        execution_results.append(result)
        execution_errors.append(error_msg)

    # Save output CSV with the required columns
    out_df = df[["question_id", "question_string", "gold_sparql", "generated_sparql"]].copy()
    out_df["execution_results"] = execution_results
    out_df["execution_error"] = execution_errors
    out_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    input_file = argv[1] 
    output_file = argv[2]
    main(input_file, output_file)