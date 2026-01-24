import json
from sys import argv
import pandas as pd
from SPARQLWrapper import SPARQLWrapper, JSON

SPARQLPATH = "http://localhost:8890/sparql"

def load_gold_answers():
    """Load gold answers from SciQA train/valid/test answer files and map id -> JSON string."""
    id_to_answer = {}
    answer_files = [
        "/Users/sherrypan/GitHub/GAR_SKGQA/datasets/sciqa/train/answers.json",
        "/Users/sherrypan/GitHub/GAR_SKGQA/datasets/sciqa/valid/answers.json",
        "/Users/sherrypan/GitHub/GAR_SKGQA/datasets/sciqa/test/answers.json",
    ]
    for path in answer_files:
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except Exception:
            continue
        for item in data.get("answers", []):
            qid = item.get("id")
            ans = item.get("answer")
            if qid is None or ans is None:
                continue
            # Normalize similarly to execution results
            if isinstance(ans, dict) and "boolean" in ans:
                normalized = {"boolean": ans.get("boolean")}
            else:
                normalized = dict(ans) if isinstance(ans, dict) else ans
                if isinstance(normalized, dict):
                    head = normalized.get("head")
                    if isinstance(head, dict):
                        head = dict(head)
                        head.pop("link", None)
                        normalized["head"] = head
                    results = normalized.get("results")
                    if isinstance(results, dict):
                        results = dict(results)
                        results.pop("distinct", None)
                        results.pop("ordered", None)
                        normalized["results"] = results
            try:
                id_to_answer[qid] = json.dumps(normalized)
            except Exception:
                try:
                    id_to_answer[qid] = json.dumps(item.get("answer"))
                except Exception:
                    id_to_answer[qid] = None
    return id_to_answer

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

def get_exact_match(gold_result, generated_result):
    """
    Compare gold SPARQL execution result with generated SPARQL execution result for exact match.
    Compares actual values, ignoring variable names and result structure differences.
    Returns 1 if they match exactly, 0 otherwise.
    """
    if gold_result is None and generated_result is None:
        return 1
    if gold_result is None or generated_result is None:
        return 0
    
    try:
        # Parse both results as JSON for comparison
        gold_parsed = json.loads(gold_result) if isinstance(gold_result, str) else gold_result
        generated_parsed = json.loads(generated_result) if isinstance(generated_result, str) else generated_result
        
        # Handle boolean results (ASK queries)
        if isinstance(gold_parsed, dict) and "boolean" in gold_parsed:
            if isinstance(generated_parsed, dict) and "boolean" in generated_parsed:
                return 1 if gold_parsed["boolean"] == generated_parsed["boolean"] else 0
            return 0
        
        # Handle binding results (SELECT queries)
        if isinstance(gold_parsed, dict) and "results" in gold_parsed:
            if not isinstance(generated_parsed, dict) or "results" not in generated_parsed:
                return 0
            
            gold_bindings = gold_parsed["results"].get("bindings", [])
            generated_bindings = generated_parsed["results"].get("bindings", [])
            
            # Compare number of results
            if len(gold_bindings) != len(generated_bindings):
                return 0
            
            # Extract and compare actual values (ignoring variable names)
            gold_values = set()
            generated_values = set()
            
            for binding in gold_bindings:
                for var_name, var_data in binding.items():
                    if isinstance(var_data, dict) and "value" in var_data:
                        gold_values.add(var_data["value"])
            
            for binding in generated_bindings:
                for var_name, var_data in binding.items():
                    if isinstance(var_data, dict) and "value" in var_data:
                        generated_values.add(var_data["value"])
            
            # Compare sets of values
            return 1 if gold_values == generated_values else 0
        
        # Fallback to direct comparison for other structures
        return 1 if gold_parsed == generated_parsed else 0
        
    except (json.JSONDecodeError, TypeError, KeyError):
        # If parsing fails, do string comparison as fallback
        return 1 if str(gold_result) == str(generated_result) else 0

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
    gold_results = []
    gold_map = load_gold_answers()
    gold_exec_results = []
    exact_matches = []
    for idx, row in df.iterrows():
        query_text = row.get("generated_sparql")
        if pd.isna(query_text) or not isinstance(query_text, str) or not query_text.strip():
            execution_results.append(None)
            execution_errors.append(None)
            gold_results.append(gold_map.get(row.get("question_id")))
            # Execute gold SPARQL when available
            gold_q = row.get("gold_sparql")
            if isinstance(gold_q, str) and gold_q.strip():
                gold_exec, _ = test(sparql_text=f"{prefixes}\n{gold_q}")
                gold_exec_results.append(gold_exec)
            else:
                gold_exec_results.append(None)
            # Compare results for exact match
            exact_matches.append(get_exact_match(gold_exec_results[-1], None))
            continue
        result, error_msg = test(sparql_text=f"{prefixes}\n{query_text}")
        print(f"Query for: {row.get('question_id')}")
        print(result)
        execution_results.append(result)
        execution_errors.append(error_msg)
        gold_results.append(gold_map.get(row.get("question_id")))
        # Execute gold SPARQL when available
        gold_q = row.get("gold_sparql")
        if isinstance(gold_q, str) and gold_q.strip():
            gold_exec, _ = test(sparql_text=f"{prefixes}\n{gold_q}")
            gold_exec_results.append(gold_exec)
        else:
            gold_exec_results.append(None)
        # Compare results for exact match
        exact_matches.append(get_exact_match(gold_exec_results[-1], result))

    # Save output CSV with the required columns
    out_df = df[["question_id", "question_string", "gold_sparql", "generated_sparql"]].copy()
    out_df["execution_results"] = execution_results
    out_df["execution_error"] = execution_errors
    out_df["gold_results"] = gold_results
    out_df["gold_sparql_query_execution"] = gold_exec_results
    out_df["exact_match"] = exact_matches
    out_df.to_csv(output_file, index=False)
    
    # Calculate and print exact match score
    total_questions = len(exact_matches)
    exact_matches_count = sum(exact_matches)
    exact_match_score = exact_matches_count / total_questions if total_questions > 0 else 0
    
    print(f"\n=== EXACT MATCH EVALUATION RESULTS ===")
    print(f"Total questions: {total_questions}")
    print(f"Exact matches: {exact_matches_count}")
    print(f"Exact match score: {exact_match_score:.4f} ({exact_match_score*100:.2f}%)")
    print(f"=====================================")

if __name__ == "__main__":
    input_file = argv[1] 
    output_file = argv[2]
    main(input_file, output_file)