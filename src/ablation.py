"""
Ablation study — unified evaluation on a single FAISS corpus.

Compares all retrieval configurations on identical conditions:
  - Same 500 validation examples
  - Same global FAISS index (all unique passages from those 500 examples)
  - Recall@1/3/5/10, MRR@10, nDCG@10 for retrieval quality
  - F1 and EM after reader integration (top-3 passages)

Configurations tested:
  1. BM25 (lexical baseline)
  2. DPR — vanilla (no fine-tuning)
  3. DPR — hard-negative fine-tuned (models/dpr_hn_v2/epoch3)

Reader uses model.generate() directly — the "text2text-generation" pipeline
task was removed in recent transformers versions.
"""

import json
import os
import string

import faiss
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DPRContextEncoder,
    DPRContextEncoderTokenizerFast,
    DPRQuestionEncoder,
    DPRQuestionEncoderTokenizerFast,
)

from load_dataset import load_hotpotqa

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VANILLA_Q_MODEL = "facebook/dpr-question_encoder-single-nq-base"
VANILLA_C_MODEL = "facebook/dpr-ctx_encoder-single-nq-base"
QA_MODEL        = "google/flan-t5-xl"

device = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize(text):
    return text.lower().translate(str.maketrans("", "", string.punctuation)).strip()


def token_f1(prediction, ground_truth):
    pred_tokens = normalize(prediction).split()
    gt_tokens   = normalize(ground_truth).split()
    common = set(pred_tokens) & set(gt_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall    = len(common) / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction, ground_truth):
    return int(normalize(prediction) == normalize(ground_truth))


def tokenize_bm25(text):
    return normalize(text).split()


def collect_passages(dataset):
    passages = []
    seen = set()
    for ex in dataset:
        for title, sentences in zip(
            ex["context"]["title"], ex["context"]["sentences"]
        ):
            if title not in seen:
                seen.add(title)
                passages.append({"title": title, "text": " ".join(sentences)})
    return passages


# ---------------------------------------------------------------------------
# BM25 retrieval
# ---------------------------------------------------------------------------

def build_bm25_index(passages):
    corpus = [tokenize_bm25(p["text"]) for p in passages]
    return BM25Okapi(corpus)


def retrieve_bm25(question, bm25_index, passages, top_k):
    scores  = bm25_index.get_scores(tokenize_bm25(question))
    indices = np.argsort(scores)[::-1][:top_k]
    return [passages[i] for i in indices]


# ---------------------------------------------------------------------------
# DPR retrieval
# ---------------------------------------------------------------------------

def load_dpr_encoders(model_dir):
    # model_dir=None -> vanilla DPR. If a path is given it MUST exist:
    # fail loudly instead of silently falling back to vanilla (the footgun
    # that once masked a broken checkpoint as a "0.43" ablation result).
    if model_dir:
        q_path = f"{model_dir}/question_encoder"
        c_path = f"{model_dir}/context_encoder"
        assert os.path.isdir(q_path), f"Fine-tuned model not found: {q_path}"
        print(f"  Loading fine-tuned DPR from {model_dir}/")
    else:
        q_path = VANILLA_Q_MODEL
        c_path = VANILLA_C_MODEL
        print("  Loading vanilla DPR.")

    q_tokenizer = DPRQuestionEncoderTokenizerFast.from_pretrained(q_path)
    c_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained(c_path)
    q_encoder   = DPRQuestionEncoder.from_pretrained(q_path).to(device).eval()
    c_encoder   = DPRContextEncoder.from_pretrained(c_path).to(device).eval()
    return q_tokenizer, q_encoder, c_tokenizer, c_encoder


def build_faiss_index(passages, c_tokenizer, c_encoder, batch_size=32):
    all_embs = []
    for i in tqdm(range(0, len(passages), batch_size), desc="  Encoding passages", leave=False):
        batch = [p["title"] + " " + p["text"] for p in passages[i:i+batch_size]]
        inputs = c_tokenizer(
            batch, return_tensors="pt",
            truncation=True, max_length=256, padding=True,
        ).to(device)
        with torch.no_grad():
            embs = c_encoder(**inputs).pooler_output.cpu().numpy()
        all_embs.append(embs)
    matrix = np.vstack(all_embs).astype("float32")
    index  = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    return index


def retrieve_dpr(question, faiss_index, passages, q_tokenizer, q_encoder, top_k):
    inputs = q_tokenizer(
        question, return_tensors="pt", truncation=True, max_length=64,
    ).to(device)
    with torch.no_grad():
        q_emb = q_encoder(**inputs).pooler_output.cpu().numpy().astype("float32")
    _, indices = faiss_index.search(q_emb, top_k)
    return [passages[i] for i in indices[0]]


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------

def read(question, top_passages, reader_model, reader_tokenizer):
    context = " ".join([p["title"] + " " + p["text"] for p in top_passages])
    context = context[:2000]
    prompt  = (
        f"Answer the question based on the context below.\n"
        f"Context: {context}\n"
        f"Question: {question}\n"
        f"Answer:"
    )
    inputs = reader_tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=1024,
    ).to(device)
    with torch.no_grad():
        output = reader_model.generate(**inputs, max_new_tokens=50, do_sample=False)
    return reader_tokenizer.decode(output[0], skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Evaluate one configuration
# ---------------------------------------------------------------------------

def evaluate_config(name, dataset, all_passages, retriever_fn, reader_model, reader_tokenizer, top_k_list, rank_k=10):
    """
    retriever_fn(question, top_k) -> list of passage dicts.

    Returns a metrics dict: Recall@k (for k in top_k_list), plus ranking
    metrics over the top rank_k (Recall@rank_k, MRR@rank_k, nDCG@rank_k),
    plus reader F1 / EM on the top-3 passages.

    HotpotQA is multi-hop: there are usually 2 gold ("supporting_facts")
    passages. Recall@k = at least one gold in the top k. MRR uses the first
    gold's rank. nDCG treats all gold passages as relevant.
    """
    recalls      = {k: 0 for k in top_k_list}
    recall_rank  = 0
    mrr_total    = 0.0
    ndcg_total   = 0.0
    f1_total     = 0.0
    em_total     = 0.0
    max_k        = max(max(top_k_list), rank_k)
    reader_k     = 3

    for ex in tqdm(dataset, desc=f"  {name}", leave=False):
        question    = ex["question"]
        answer      = ex["answer"]
        gold_titles = set(ex["supporting_facts"]["title"])

        # Retrieve max_k passages once
        top_passages  = retriever_fn(question, max_k)
        ranked_titles = [p["title"] for p in top_passages]

        # Recall@k (at least one gold in top k)
        for k in top_k_list:
            if gold_titles & set(ranked_titles[:k]):
                recalls[k] += 1
        if gold_titles & set(ranked_titles[:rank_k]):
            recall_rank += 1

        # MRR@rank_k (reciprocal rank of the first gold passage)
        for rank, title in enumerate(ranked_titles[:rank_k], start=1):
            if title in gold_titles:
                mrr_total += 1.0 / rank
                break

        # nDCG@rank_k (all gold passages relevant)
        dcg  = sum(1.0 / np.log2(i + 2)
                   for i, t in enumerate(ranked_titles[:rank_k]) if t in gold_titles)
        idcg = sum(1.0 / np.log2(i + 2)
                   for i in range(min(len(gold_titles), rank_k)))
        ndcg_total += dcg / idcg if idcg > 0 else 0.0

        # Reader on top reader_k passages
        predicted = read(question, top_passages[:reader_k], reader_model, reader_tokenizer)
        f1_total += token_f1(predicted, answer)
        em_total += exact_match(predicted, answer)

    n = len(dataset)
    metrics = {"config": name, "n": n}
    for k in top_k_list:
        metrics[f"Recall@{k}"] = round(recalls[k] / n, 4)
    metrics[f"Recall@{rank_k}"] = round(recall_rank / n, 4)
    metrics[f"MRR@{rank_k}"]    = round(mrr_total / n, 4)
    metrics[f"nDCG@{rank_k}"]   = round(ndcg_total / n, 4)
    metrics["F1"]          = round(f1_total / n, 4)
    metrics["Exact Match"] = round(em_total / n, 4)
    return metrics


# ---------------------------------------------------------------------------
# Main ablation runner
# ---------------------------------------------------------------------------

def run_ablation(
    max_samples   = 500,
    top_k_list    = [1, 3, 5],
    dpr_model_dirs = None,
    output_path   = "results/ablation_results.json",
):
    if dpr_model_dirs is None:
        dpr_model_dirs = {
            "DPR (vanilla)": None,
            "DPR + HN":      "models/dpr_hn_v2/epoch3",
        }

    print(f"Device: {device}")
    os.makedirs("results", exist_ok=True)

    dataset     = load_hotpotqa(split="validation", max_samples=max_samples)
    all_passages = collect_passages(dataset)
    print(f"Corpus: {len(all_passages)} unique passages")

    # Load reader once — shared across all configs.
    # Use model.generate() directly: the "text2text-generation" pipeline task
    # was removed in recent transformers versions.
    print(f"Loading QA reader ({QA_MODEL})...")
    reader_tokenizer = AutoTokenizer.from_pretrained(QA_MODEL)
    reader_model     = AutoModelForSeq2SeqLM.from_pretrained(QA_MODEL).to(device).eval()

    all_metrics = []

    # ------------------------------------------------------------------
    # 1. BM25
    # ------------------------------------------------------------------
    print(f"\n[1/{1 + len(dpr_model_dirs)}] BM25")
    bm25_index = build_bm25_index(all_passages)
    bm25_fn    = lambda q, k: retrieve_bm25(q, bm25_index, all_passages, k)
    metrics    = evaluate_config("BM25", dataset, all_passages, bm25_fn, reader_model, reader_tokenizer, top_k_list)
    all_metrics.append(metrics)
    print(f"  → {metrics}")

    # ------------------------------------------------------------------
    # DPR variants (vanilla + fine-tuned)
    # ------------------------------------------------------------------
    for idx, (name, model_dir) in enumerate(dpr_model_dirs.items(), start=2):
        print(f"\n[{idx}/{1+len(dpr_model_dirs)}] {name}")
        q_tok, q_enc, c_tok, c_enc = load_dpr_encoders(model_dir)
        faiss_idx = build_faiss_index(all_passages, c_tok, c_enc)
        dpr_fn    = lambda q, k, qt=q_tok, qe=q_enc: retrieve_dpr(
            q, faiss_idx, all_passages, qt, qe, k
        )
        metrics = evaluate_config(name, dataset, all_passages, dpr_fn, reader_model, reader_tokenizer, top_k_list)
        all_metrics.append(metrics)
        print(f"  → {metrics}")

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    print("\n=== Ablation Results ===")
    rk    = 10
    cols  = [f"Recall@{k}" for k in top_k_list] + [f"Recall@{rk}", f"MRR@{rk}", f"nDCG@{rk}", "F1", "Exact Match"]
    short = {**{f"Recall@{k}": f"R@{k}" for k in top_k_list},
             f"Recall@{rk}": f"R@{rk}", f"MRR@{rk}": f"MRR@{rk}",
             f"nDCG@{rk}": f"nDCG@{rk}", "F1": "F1", "Exact Match": "EM"}
    header = f"{'Config':<18}" + "".join(f"{short[c]:>9}" for c in cols)
    print(header)
    print("-" * len(header))
    for m in all_metrics:
        print(f"{m['config']:<18}" + "".join(f"{m[c]:>9.4f}" for c in cols))

    with open(output_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSaved to {output_path}")
    return all_metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_ablation(
        max_samples    = 500,
        top_k_list     = [1, 3, 5],
        dpr_model_dirs = {
            "DPR (vanilla)": None,
            "DPR + HN":      "models/dpr_hn_v2/epoch3",
        },
        output_path = "results/ablation_results.json",
    )
