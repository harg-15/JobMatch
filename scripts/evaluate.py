"""Compare keyword search, vector search and vector + LLM re-rank.

    python -m scripts.evaluate                      # uses eval/labels.csv
    python -m scripts.evaluate --labels my_labels.csv

labels.csv has one row per (resume, job) pair that is a GOOD match:

    resume,job_external_id
    backend_student.txt,sj-001

Resumes are plain text files in eval/resumes/. Any job not listed for a
resume counts as not relevant.

Metrics, averaged over resumes:
  precision@k  - of the k jobs shown, how many were good matches
  recall@k     - of all good matches, how many made it into the top k
"""
import argparse
import csv
import statistics
import time
from collections import defaultdict
from pathlib import Path

from app import db
from app.embeddings import get_embedder
from app.matcher import match_resume
from app.rerank import openai_compatible_llm

EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"


def load_labels(path: Path) -> dict[str, set[str]]:
    labels: dict[str, set[str]] = defaultdict(set)
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels[row["resume"].strip()].add(row["job_external_id"].strip())
    return labels


def evaluate(conn, embedder, llm, labels, resumes_dir: Path, modes, k: int = 5):
    results = {}
    for mode in modes:
        precisions, recalls, latencies = [], [], []
        for resume_file, relevant in labels.items():
            text = (resumes_dir / resume_file).read_text(encoding="utf-8")
            start = time.perf_counter()
            matches, _ = match_resume(conn, embedder, llm, text, mode=mode, top_k=k)
            latencies.append(time.perf_counter() - start)

            shown = [m["external_id"] for m in matches]
            hits = len(set(shown) & relevant)
            precisions.append(hits / k)
            recalls.append(hits / len(relevant))
        results[mode] = {
            "precision": statistics.mean(precisions),
            "recall": statistics.mean(recalls),
            "median_latency": statistics.median(latencies),
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, default=EVAL_DIR / "labels.csv")
    parser.add_argument("--resumes", type=Path, default=EVAL_DIR / "resumes")
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()

    labels = load_labels(args.labels)
    llm = openai_compatible_llm()
    modes = ["keyword", "vector"] + (["rerank"] if llm else [])
    if not llm:
        print("LLM_API_KEY / LLM_MODEL not set - skipping the re-rank mode\n")

    with db.connect() as conn:
        embedder = get_embedder()
        embedder.embed_query("warm up")  # don't count model loading in latency
        results = evaluate(conn, embedder, llm, labels, args.resumes, modes, args.k)

    print(f"{len(labels)} resumes, k={args.k}\n")
    print(f"{'mode':<10}{'precision@k':>13}{'recall@k':>11}{'median latency':>17}")
    for mode, r in results.items():
        print(f"{mode:<10}{r['precision']:>13.2f}{r['recall']:>11.2f}"
              f"{r['median_latency']:>15.2f} s")


if __name__ == "__main__":
    main()
