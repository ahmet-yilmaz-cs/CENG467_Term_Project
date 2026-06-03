"""
Hard-negative fine-tuning of DPR for HotpotQA retrieval.

Recipe (validated: beats vanilla DPR 0.714 -> 0.844 R@1 on global retrieval):
  - in-batch + hard-negative contrastive loss, curriculum HN schedule [1, 3, 5]
  - AdamW lr 2e-5, linear warmup, gradient clipping, batch_size 16

Each epoch is saved to its own directory (output_dir/epoch{N}); the final
checkpoint is chosen by GLOBAL retrieval via src/ablation.py — NOT by the
mini-pool recall below, which is a misleading, near-saturated signal.
"""

import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import (
    DPRContextEncoder,
    DPRContextEncoderTokenizer,
    DPRQuestionEncoder,
    DPRQuestionEncoderTokenizer,
    get_linear_schedule_with_warmup,
)


QUESTION_MODEL = "facebook/dpr-question_encoder-single-nq-base"
CONTEXT_MODEL = "facebook/dpr-ctx_encoder-single-nq-base"

device = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_hard_negatives(path="data/hard_negatives_train.json", max_samples=None):
    with open(path) as f:
        data = json.load(f)
    if max_samples:
        data = data[:max_samples]
    print(f"Loaded {len(data)} training examples from {path}")
    return data


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def contrastive_loss(q_embs, p_embs, batch_size):
    """
    q_embs : (B, D)   — question embeddings
    p_embs : (B + HN, D) — positives first, then all hard negatives concatenated

    For question i, the positive is p_embs[i].
    All other passages (in-batch positives + hard negatives) act as negatives.
    Target for each question is its diagonal index.
    """
    scores = torch.mm(q_embs, p_embs.t())          # (B, B + HN)
    targets = torch.arange(batch_size, device=device)
    return F.cross_entropy(scores, targets)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    data_path="data/hard_negatives_train.json",
    output_dir="models/dpr_hn_v2",
    max_samples=None,
    epochs=3,
    batch_size=16,
    lr=2e-5,
    recall_eval_samples=200,
    resume_from=None,
    val_path="data/hard_negatives_val.json",
):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Device: {device}")

    # Load encoders — from checkpoint if resume_from is given, else from pretrained
    if resume_from is not None:
        print(f"Resuming from checkpoint: {resume_from}")
        q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(f"{resume_from}/question_encoder")
        q_encoder   = DPRQuestionEncoder.from_pretrained(f"{resume_from}/question_encoder").to(device)
        c_tokenizer = DPRContextEncoderTokenizer.from_pretrained(f"{resume_from}/context_encoder")
        c_encoder   = DPRContextEncoder.from_pretrained(f"{resume_from}/context_encoder").to(device)
    else:
        print("Loading encoders...")
        q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(QUESTION_MODEL)
        q_encoder   = DPRQuestionEncoder.from_pretrained(QUESTION_MODEL).to(device)
        c_tokenizer = DPRContextEncoderTokenizer.from_pretrained(CONTEXT_MODEL)
        c_encoder   = DPRContextEncoder.from_pretrained(CONTEXT_MODEL).to(device)

    q_encoder.train()
    c_encoder.train()

    optimizer = torch.optim.AdamW(
        list(q_encoder.parameters()) + list(c_encoder.parameters()),
        lr=lr,
    )

    data = load_hard_negatives(data_path, max_samples=max_samples)

    # LR scheduler with linear warmup (10% of total steps)
    total_steps    = (len(data) // batch_size) * epochs
    num_warmup     = max(1, total_steps // 10)
    scheduler      = get_linear_schedule_with_warmup(optimizer, num_warmup, total_steps)

    # Curriculum: number of hard negatives to use per epoch
    # Epoch 1 → 1 HN (easy), Epoch 2 → 3 HN, Epoch 3 → 5 HN (full)
    hn_schedule = [1, 3, 5]

    for epoch in range(epochs):
        hn_count   = hn_schedule[min(epoch, len(hn_schedule) - 1)]
        total_loss = 0.0
        steps      = 0
        print(f"\nEpoch {epoch + 1}/{epochs}  |  curriculum hard negatives: {hn_count}")

        for start in tqdm(range(0, len(data), batch_size), desc=f"Epoch {epoch + 1}/{epochs}"):
            batch     = data[start : start + batch_size]
            actual_bs = len(batch)

            questions = [ex["question"] for ex in batch]

            # First positive passage per question
            positives = [
                ex["positive_passages"][0]["title"] + " " + ex["positive_passages"][0]["text"]
                for ex in batch
            ]

            # Curriculum: use hn_count hard negatives per question this epoch
            hard_negs = [
                p["title"] + " " + p["text"]
                for ex in batch
                for p in ex["hard_negatives"][:hn_count]
            ]

            all_passages = positives + hard_negs

            q_inputs = q_tokenizer(
                questions, return_tensors="pt",
                truncation=True, max_length=64, padding=True,
            ).to(device)

            p_inputs = c_tokenizer(
                all_passages, return_tensors="pt",
                truncation=True, max_length=256, padding=True,
            ).to(device)

            q_embs = q_encoder(**q_inputs).pooler_output   # (B, 768)
            p_embs = c_encoder(**p_inputs).pooler_output   # (B + HN, 768)

            loss = contrastive_loss(q_embs, p_embs, actual_bs)

            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping prevents gradient explosion with hard negatives
            torch.nn.utils.clip_grad_norm_(
                list(q_encoder.parameters()) + list(c_encoder.parameters()),
                max_norm=1.0,
            )
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            steps      += 1

        avg_loss = total_loss / steps
        print(f"  avg loss: {avg_loss:.4f}  |  lr: {scheduler.get_last_lr()[0]:.2e}")

        # Save THIS epoch to its own directory (never overwrite).
        # Final selection is done on GLOBAL retrieval via ablation.py.
        epoch_dir = os.path.join(output_dir, f"epoch{epoch + 1}")
        q_encoder.save_pretrained(f"{epoch_dir}/question_encoder")
        q_tokenizer.save_pretrained(f"{epoch_dir}/question_encoder")
        c_encoder.save_pretrained(f"{epoch_dir}/context_encoder")
        c_tokenizer.save_pretrained(f"{epoch_dir}/context_encoder")
        print(f"  ✓ epoch {epoch + 1} saved to {epoch_dir}/")

        # Mini-pool recall — DIAGNOSTIC ONLY, not used to pick the final model.
        evaluate_recall(model_dir=epoch_dir, max_samples=recall_eval_samples, val_path=val_path)

    print(f"\nTraining complete. Per-epoch checkpoints under {output_dir}/")
    print("Pick the best epoch with src/ablation.py (global retrieval).")
    return output_dir


# ---------------------------------------------------------------------------
# Evaluation (Recall@5) using fine-tuned model
# ---------------------------------------------------------------------------

def evaluate_recall(model_dir="models/dpr_hn_v2", max_samples=200,
                    val_path="data/hard_negatives_val.json"):
    """
    Mini-pool retrieval sanity check on the held-out hard-negatives set: each
    question is ranked only against its own positives + hard negatives (~7
    candidates). DIAGNOSTIC ONLY — the pool is tiny and near-saturated, so MRR
    here is NOT a reliable signal for picking the final checkpoint (it once
    selected a checkpoint that scored only 0.43 R@1 on real global retrieval).

    Pick the final epoch with ablation.py, which evaluates on the full pooled
    corpus of the HotpotQA validation split (true global retrieval).

    Returns a dict with Recall@1, Recall@5, Recall@10, MRR@10, nDCG@10.
    """
    print(f"\nEvaluating {model_dir} on {val_path} (max {max_samples} samples)...")

    q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(f"{model_dir}/question_encoder")
    q_encoder   = DPRQuestionEncoder.from_pretrained(f"{model_dir}/question_encoder").to(device).eval()
    c_tokenizer = DPRContextEncoderTokenizer.from_pretrained(f"{model_dir}/context_encoder")
    c_encoder   = DPRContextEncoder.from_pretrained(f"{model_dir}/context_encoder").to(device).eval()

    with open(val_path) as f:
        dataset = json.load(f)
    if max_samples:
        dataset = dataset[:max_samples]

    recall_counts = {1: 0, 5: 0, 10: 0}
    mrr_sum       = 0.0
    ndcg_sum      = 0.0
    top_k_max     = 10

    for ex in tqdm(dataset, desc="Retrieval eval"):
        question    = ex["question"]
        gold_titles = set(p["title"] for p in ex["positive_passages"])

        passages = ex["positive_passages"] + ex["hard_negatives"]

        q_inputs = q_tokenizer(
            question, return_tensors="pt", truncation=True, max_length=64,
        ).to(device)
        with torch.no_grad():
            q_emb = q_encoder(**q_inputs).pooler_output.squeeze(0).cpu().numpy()

        p_texts  = [p["title"] + " " + p["text"] for p in passages]
        p_inputs = c_tokenizer(
            p_texts, return_tensors="pt",
            truncation=True, max_length=256, padding=True,
        ).to(device)
        with torch.no_grad():
            p_embs = c_encoder(**p_inputs).pooler_output.cpu().numpy()

        scores        = p_embs @ q_emb
        top_indices   = np.argsort(scores)[::-1][:top_k_max]
        ranked_titles = [passages[i]["title"] for i in top_indices]

        # Recall@k
        for k in recall_counts:
            if gold_titles & set(ranked_titles[:k]):
                recall_counts[k] += 1

        # MRR@10
        for rank, title in enumerate(ranked_titles, start=1):
            if title in gold_titles:
                mrr_sum += 1.0 / rank
                break

        # nDCG@10
        dcg  = sum(
            1.0 / np.log2(i + 2)
            for i, t in enumerate(ranked_titles)
            if t in gold_titles
        )
        idcg = sum(
            1.0 / np.log2(i + 2)
            for i in range(min(len(gold_titles), top_k_max))
        )
        ndcg_sum += dcg / idcg if idcg > 0 else 0.0

    n = len(dataset)
    metrics = {
        "Recall@1":  round(recall_counts[1]  / n, 4),
        "Recall@5":  round(recall_counts[5]  / n, 4),
        "Recall@10": round(recall_counts[10] / n, 4),
        "MRR@10":    round(mrr_sum  / n, 4),
        "nDCG@10":   round(ndcg_sum / n, 4),
    }

    print(f"\n=== Retrieval Metrics ({val_path}, n={n}) ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    os.makedirs("results", exist_ok=True)
    out_path = "results/dpr_hn_results.json"
    with open(out_path, "w") as f:
        json.dump({"model_dir": model_dir, "n": n, "metrics": metrics}, f, indent=2)
    print(f"  Saved to {out_path}")
    return metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    output_dir = train(
        data_path="data/hard_negatives_train.json",
        output_dir="models/dpr_hn_v2",
        max_samples=None,   # local test: set small; full run: None
        epochs=3,
        batch_size=16,      # small GPU: 8; CPU: 4
        lr=2e-5,
    )
    print("\nNext: select the best epoch on global retrieval with src/ablation.py")
    print(f"      (point 'DPR + HN' at {output_dir}/epoch3 or whichever epoch wins).")
