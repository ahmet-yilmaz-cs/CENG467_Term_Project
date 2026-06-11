"""
Statistical analysis of the ablation.

Runs a detailed evaluation that records *per-example* F1, EM and the retrieved
passage titles for every configuration, then reports:
  - 95% bootstrap confidence intervals on F1, EM, Recall@1/5, strict Recall@5
  - paired-bootstrap significance of each fine-tuned config vs vanilla DPR
  - strict (both-supporting-passage) recall for the multi-hop setting
  - a per-category breakdown (bridge / comparison / yes-no / span)

The detailed records are cached to results/eval_detail.json so the statistics
(report_statistics) can be re-run without re-evaluating the models.

Shared retrieval/reader helpers are imported from ablation.py.
Run from the src/ directory:  python analysis.py
"""
import json
import os

import numpy as np
import torch
from tqdm import tqdm

from load_dataset import load_hotpotqa
from ablation import (
    normalize,
    token_f1,
    exact_match,
    collect_passages,
    build_bm25_index,
    retrieve_bm25,
    load_dpr_encoders,
    build_faiss_index,
    retrieve_dpr,
    read,
    QA_MODEL,
    device,
)
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Detailed evaluation (per-example records)
# ---------------------------------------------------------------------------

def run_detailed_eval(
    max_samples=500,
    top_k_ret=10,
    reader_k=3,
    dpr_model_dirs=None,
    output_path="results/eval_detail.json",
):
    """Evaluate every config and store one record per question."""
    if dpr_model_dirs is None:
        dpr_model_dirs = {"DPR (vanilla)": None, "DPR + HN": "models/dpr_hn_v2/epoch3"}

    os.makedirs("results", exist_ok=True)
    dataset  = load_hotpotqa(split="validation", max_samples=max_samples)
    passages = collect_passages(dataset)
    print(f"Corpus: {len(passages)} unique passages")

    # Shared Flan-T5-xl reader (same prompt/budget as the main ablation).
    print(f"Loading reader ({QA_MODEL})...")
    reader_tokenizer = AutoTokenizer.from_pretrained(QA_MODEL)
    reader_model     = AutoModelForSeq2SeqLM.from_pretrained(QA_MODEL).to(device).eval()

    # One retriever per config (top_k_ret kept so strict recall can use depth>k).
    retrievers = {"BM25": None}
    bm25_index = build_bm25_index(passages)
    retrievers["BM25"] = lambda q: retrieve_bm25(q, bm25_index, passages, top_k_ret)
    for name, model_dir in dpr_model_dirs.items():
        q_tok, q_enc, c_tok, c_enc = load_dpr_encoders(model_dir)
        faiss_idx = build_faiss_index(passages, c_tok, c_enc)
        retrievers[name] = (
            lambda q, qt=q_tok, qe=q_enc, idx=faiss_idx:
            retrieve_dpr(q, idx, passages, qt, qe, top_k_ret)
        )

    records = []
    for ex in tqdm(dataset, desc="Detailed eval"):
        question = ex["question"]
        answer   = ex["answer"]
        gold     = list(set(ex["supporting_facts"]["title"]))
        record = {
            "type":  ex.get("type", ""),
            "level": ex.get("level", ""),
            "yesno": int(normalize(answer) in ("yes", "no")),
            "n_gold": len(gold),
            "configs": {},
        }
        for name, retrieve_fn in retrievers.items():
            top = retrieve_fn(question)
            pred = read(question, top[:reader_k], reader_model, reader_tokenizer)
            record["configs"][name] = {
                "f1": token_f1(pred, answer),
                "em": exact_match(pred, answer),
                "titles": [p["title"] for p in top],
                "gold": gold,
            }
        records.append(record)

    with open(output_path, "w") as f:
        json.dump(records, f)
    print(f"Saved {len(records)} per-example records to {output_path}")
    return records


# ---------------------------------------------------------------------------
# Statistics (pure post-processing on the saved records)
# ---------------------------------------------------------------------------

def _bootstrap_ci(values, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    means = [rng.choice(values, len(values)).mean() for _ in range(n_boot)]
    return values.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def _recall(record, config, k, strict=False):
    gold = set(record["configs"][config]["gold"])
    topk = set(record["configs"][config]["titles"][:k])
    return int(gold <= topk) if strict else int(bool(gold & topk))


def report_statistics(records, configs=None, baseline="DPR (vanilla)"):
    if configs is None:
        configs = list(records[0]["configs"].keys())

    print("\n=== Metrics with 95% bootstrap CI ===")
    for config in configs:
        metrics = {
            "F1":       [r["configs"][config]["f1"] for r in records],
            "EM":       [r["configs"][config]["em"] for r in records],
            "R@1":      [_recall(r, config, 1) for r in records],
            "R@5 any":  [_recall(r, config, 5) for r in records],
            "R@5 both": [_recall(r, config, 5, strict=True) for r in records],
        }
        for label, values in metrics.items():
            mean, lo, hi = _bootstrap_ci(values)
            print(f"  {config:<14} {label:<9}: {mean:.3f}  [{lo:.3f}, {hi:.3f}]")
        print()

    if baseline in configs:
        print(f"=== Significance vs {baseline} (paired bootstrap, 5000 resamples) ===")
        rng = np.random.default_rng(0)
        for config in configs:
            if config in ("BM25", baseline):
                continue
            for label, value_fn in [("F1", lambda r, c: r["configs"][c]["f1"]),
                                    ("R@1", lambda r, c: _recall(r, c, 1))]:
                diff = np.array([value_fn(r, config) - value_fn(r, baseline) for r in records])
                boot = [rng.choice(diff, len(diff)).mean() for _ in range(5000)]
                p = 2 * min(np.mean(np.asarray(boot) <= 0), np.mean(np.asarray(boot) >= 0))
                print(f"  {config:<10} Δ{label:<3} = {diff.mean():+.3f}  "
                      f"95% CI [{np.percentile(boot, 2.5):+.3f}, {np.percentile(boot, 97.5):+.3f}]  "
                      f"p ≈ {p:.4f}")
        print()

    target = configs[-1]
    print(f"=== {target} breakdown by question category (n, F1, EM, R@1) ===")
    categories = [
        ("bridge",     lambda r: r["type"] == "bridge"),
        ("comparison", lambda r: r["type"] == "comparison"),
        ("yes/no",     lambda r: r["yesno"] == 1),
        ("span",       lambda r: r["yesno"] == 0),
    ]
    for label, mask in categories:
        subset = [r for r in records if mask(r)]
        if not subset:
            continue
        f1  = np.mean([r["configs"][target]["f1"] for r in subset])
        em  = np.mean([r["configs"][target]["em"] for r in subset])
        r1  = np.mean([_recall(r, target, 1) for r in subset])
        print(f"  {label:<12} n={len(subset):<4} F1 {f1:.3f}  EM {em:.3f}  R@1 {r1:.3f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    detail_path = "results/eval_detail.json"
    if os.path.exists(detail_path):
        print(f"Loading cached records from {detail_path}")
        records = json.load(open(detail_path))
    else:
        records = run_detailed_eval(max_samples=500, output_path=detail_path)
    report_statistics(records)
