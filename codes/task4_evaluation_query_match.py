import argparse
import pandas as pd
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def compute_bleu(reference: str, hypothesis: str) -> float:
    ref = reference or ""
    hyp = hypothesis or ""
    if not ref.strip() or not hyp.strip():
        return 0.0
    smooth = SmoothingFunction().method1
    return float(sentence_bleu([ref.split()], hyp.split(), weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth))


def compute_rouge(reference: str, hypothesis: str) -> dict:
    ref = reference or ""
    hyp = hypothesis or ""
    if not ref.strip() or not hyp.strip():
        return {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
    return Rouge().get_scores(hyp, ref, avg=True)


def compute_exact_match(reference: str, hypothesis: str) -> float:
    return 1.0 if (reference or "") == (hypothesis or "") else 0.0


def evaluate_file(input_csv: str, output_csv: str) -> dict:
    df = pd.read_csv(input_csv)
    required = {"question_id", "question_string", "gold_sparql", "generated_sparql", "error"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {input_csv}: {sorted(list(missing))}")

    scores = []
    for _, row in df.iterrows():
        gold = row.get("gold_sparql", "") or ""
        pred = row.get("generated_sparql", "") or ""

        bleu = compute_bleu(gold, pred)
        rouge_all = compute_rouge(gold, pred)
        em = compute_exact_match(gold, pred)

        entry = {
            "question_id": row.get("question_id"),
            "question_string": row.get("question_string"),
            "gold_sparql": gold,
            "generated_sparql": pred,
            "bleu": bleu,
            "rouge1_f1": float(rouge_all["rouge-1"]["f"]),
            "rouge2_f1": float(rouge_all["rouge-2"]["f"]),
            "rougel_f1": float(rouge_all["rouge-l"]["f"]),
            "exact_match": em,
            "error": row.get("error", "")
        }
        scores.append(entry)

    out_df = pd.DataFrame(scores)
    out_df.to_csv(output_csv, index=False)

    # Aggregates
    aggregates = {
        "bleu": float(out_df["bleu"].mean()) if not out_df.empty else 0.0,
        "rouge1_f1": float(out_df["rouge1_f1"].mean()) if not out_df.empty else 0.0,
        "rouge2_f1": float(out_df["rouge2_f1"].mean()) if not out_df.empty else 0.0,
        "rougel_f1": float(out_df["rougel_f1"].mean()) if not out_df.empty else 0.0,
        "exact_match": float(out_df["exact_match"].mean()) if not out_df.empty else 0.0,
        "count": int(len(out_df))
    }
    return aggregates


def main():
    parser = argparse.ArgumentParser(description="Task4: Evaluate generated SPARQL vs gold using BLEU, ROUGE, Exact Match")
    parser.add_argument("--input_csv", required=True, help="Task3 output CSV with columns: question_id,question_string,gold_sparql,generated_sparql,error")
    parser.add_argument("--output_csv", required=True, help="Where to write per-row scores CSV")
    args = parser.parse_args()

    metrics = evaluate_file(args.input_csv, args.output_csv)
    print("Evaluation summary:")
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()


