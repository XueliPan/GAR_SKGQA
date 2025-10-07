import re
import csv
import json
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge


def mask(txt: str) -> str:
    """Replace SPARQL variable names like ?x or ?person with ?MASKED."""
    if txt is None:
        return ""
    return re.sub(r'\?\w+', '?MASKED', txt)


def format_text(txt: str) -> str:
    """Normalize text: lowercase, remove punctuation, articles, and extra spaces."""
    if txt is None:
        return ""
    RE_ART = re.compile(r'\b(a|an|the)\b')
    RE_PUNC = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')
    lower = txt.lower()
    no_punc = RE_PUNC.sub(' ', lower)
    no_art = RE_ART.sub(' ', no_punc)
    return ' '.join(no_art.split())


def rouge_score(rog_scores: List[Dict[str, Any]]) -> Tuple[float, float, float]:
    """Average rouge-1, rouge-2, rouge-l F1-scores."""
    if not rog_scores:
        return 0.0, 0.0, 0.0

    r1 = sum([s.get("rouge-1", {}).get("f", 0.0) for s in rog_scores])
    r2 = sum([s.get("rouge-2", {}).get("f", 0.0) for s in rog_scores])
    rl = sum([s.get("rouge-l", {}).get("f", 0.0) for s in rog_scores])
    n = len(rog_scores)
    return r1 / n, r2 / n, rl / n


def metric_em(preds: List[str], golds: List[str], language: str = "sparql") -> List[int]:
    """Compute Exact Match per example and return a list of 1/0 values."""
    # use mask before exact match calculation
    results = []
    for pred, gold in zip(preds, golds):
        if language == "sparql":
            print("masking pred and gold.....")
            pred = mask(pred)
            gold = mask(gold)
            print(f"masked pred:\n{pred}\nmasked gold:\n{gold}")
        pred_norm = " ".join(pred.split()).strip()
        gold_norm = " ".join(gold.split()).strip()
        results.append(int(pred_norm == gold_norm))
    return results


def evaluate(pred: str, gold: str) -> Tuple[float, float, float, float, Dict[str, Any], float]:
    """Compute all metrics for one prediction-gold pair."""
    pred_tokens = pred.split()
    gold_tokens = gold.split()

    # Precision, recall, F1
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    precision = num_same / len(pred_tokens) if pred_tokens else 0
    recall = num_same / len(gold_tokens) if gold_tokens else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0

    # ROUGE
    rouge = Rouge()
    ro = rouge.get_scores(gold, pred, avg=True)

    # BLEU scores
    smooth = SmoothingFunction().method1
    bleu_c = sentence_bleu([gold_tokens], pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)
    bleu_4 = sentence_bleu([gold_tokens], pred_tokens, weights=(0, 0, 0, 1), smoothing_function=smooth)

    return precision, recall, f1, bleu_c, ro, bleu_4


def safe_mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate_csv(input_csv: str, output_csv: str, summary_csv: str, language: str = "sparql"):
    """Compute metrics per row and output both detailed and summary CSVs."""
    preds, golds, qids, questions = [], [], [], []

    # Read input CSV
    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qids.append(row["question_id"])
            questions.append(row["question_string"])
            golds.append(row["gold_sparql"])
            preds.append(row["generated_sparql"])
    print("Finished reading input CSV...")
    preds_fmt = [format_text(mask(p)) if language == "sparql" else format_text(p) for p in preds]
    golds_fmt = [format_text(mask(g)) if language == "sparql" else format_text(g) for g in golds]
    print(f"Evaluating {len(preds)} predictions...")
    print(f"Sample prediction:\n{preds[0]}\nSample gold:\n{golds[0]}")

    print("Calculating Exact Match...")
    em_scores = metric_em(preds, golds, language)
    print(f"Exact Match scores:\n{em_scores}")

    # Lists for aggregating average metrics
    precisions, recalls, f1s, bleu_cs, rouges, bleu4s = [], [], [], [], [], []

    # Prepare CSV writer for detailed results
    with open(output_csv, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.writer(f_out)
        writer.writerow([
            "question_id", "question_string", "gold_sparql", "generated_sparql",
            "Exact_Match", "Precision", "Recall", "F1",
            "BLEU_Cumulative", "BLEU_4",
            "ROUGE_1", "ROUGE_2", "ROUGE_L"
        ])

        for i, (p, g) in enumerate(zip(preds_fmt, golds_fmt)):
            precision, recall, f1, bleu_c, ro, bleu_4 = evaluate(p, g)
            rouges.append(ro)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            bleu_cs.append(bleu_c)
            bleu4s.append(bleu_4)
            r1, r2, rl = ro["rouge-1"]["f"], ro["rouge-2"]["f"], ro["rouge-l"]["f"]

            writer.writerow([
                qids[i], questions[i], golds[i], preds[i],
                em_scores[i], precision, recall, f1,
                bleu_c, bleu_4, r1, r2, rl
            ])

    # Aggregate averages
    rouge_1_avg, rouge_2_avg, rouge_l_avg = rouge_score(rouges)

    summary = {
        "Exact_Match": safe_mean(em_scores),
        "Precision": safe_mean(precisions),
        "Recall": safe_mean(recalls),
        "F1_Score": safe_mean(f1s),
        "BLEU_Cumulative": safe_mean(bleu_cs),
        "BLEU_4": safe_mean(bleu4s),
        "ROUGE_1": rouge_1_avg,
        "ROUGE_2": rouge_2_avg,
        "ROUGE_L": rouge_l_avg
    }

    # Write summary
    with open(summary_csv, "w", newline="", encoding="utf-8") as f_sum:
        writer = csv.writer(f_sum)
        writer.writerow(["Metric", "Average_Score"])
        for k, v in summary.items():
            writer.writerow([k, v])

    print(f"✅ Detailed results saved to: {output_csv}")
    print(f"✅ Summary averages saved to: {summary_csv}")


if __name__ == "__main__":
    # Example usage
    evaluate_csv(
        input_csv="results/hop1_oneshot_candidate_generated_sparql_details.csv",
        output_csv="results/hop1_oneshot_candidate_query_match_metrics_details.csv",
        summary_csv="results/hop1_oneshot_candidate_query_match_metrics_summary.csv",
        language="sparql"
    )
