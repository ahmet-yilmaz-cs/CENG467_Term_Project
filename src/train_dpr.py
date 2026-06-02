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

from load_dataset import load_hotpotqa

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
    output_dir="models/dpr_finetuned",
    max_samples=None,
    epochs=3,
    batch_size=4,
    lr=2e-5,
    recall_eval_samples=200,
    resume_from=None,
):
    best_dir = os.path.join(output_dir, "best_model")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(best_dir, exist_ok=True)
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

    best_score = 0.0   # tracked via MRR@10

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

        # Save current epoch checkpoint, evaluate on held-out val, keep best
        q_encoder.save_pretrained(f"{output_dir}/question_encoder")
        q_tokenizer.save_pretrained(f"{output_dir}/question_encoder")
        c_encoder.save_pretrained(f"{output_dir}/context_encoder")
        c_tokenizer.save_pretrained(f"{output_dir}/context_encoder")

        metrics = evaluate_recall(model_dir=output_dir, max_samples=recall_eval_samples)

        if metrics["MRR@10"] > best_score:
            best_score = metrics["MRR@10"]
            q_encoder.save_pretrained(f"{best_dir}/question_encoder")
            q_tokenizer.save_pretrained(f"{best_dir}/question_encoder")
            c_encoder.save_pretrained(f"{best_dir}/context_encoder")
            c_tokenizer.save_pretrained(f"{best_dir}/context_encoder")
            print(f"  ✓ Best model saved (MRR@10: {best_score:.4f})")

    print(f"\nTraining complete. Best MRR@10: {best_score:.4f}")
    print(f"Best model saved to {best_dir}/")
    return best_dir


# ---------------------------------------------------------------------------
# Evaluation (Recall@5) using fine-tuned model
# ---------------------------------------------------------------------------

def evaluate_recall(model_dir="models/dpr_finetuned", max_samples=200,
                    split="train", start=800):
    """
    Evaluates retrieval quality on a held-out slice of the HotpotQA train split
    (default: examples 800-999).  This set was never used for training, so it
    provides an unbiased signal for checkpoint selection.

    Final ablation evaluation is performed separately on the HotpotQA
    *validation* split (ablation.py), which is completely independent.

    Returns a dict with Recall@1, Recall@5, Recall@10, MRR@10, nDCG@10.
    """
    print(f"\nEvaluating {model_dir} on {split}[{start}:{start + max_samples}]...")

    q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(f"{model_dir}/question_encoder")
    q_encoder   = DPRQuestionEncoder.from_pretrained(f"{model_dir}/question_encoder").to(device).eval()
    c_tokenizer = DPRContextEncoderTokenizer.from_pretrained(f"{model_dir}/context_encoder")
    c_encoder   = DPRContextEncoder.from_pretrained(f"{model_dir}/context_encoder").to(device).eval()

    dataset = load_hotpotqa(split=split, max_samples=max_samples, start=start)

    recall_counts = {1: 0, 5: 0, 10: 0}
    mrr_sum       = 0.0
    ndcg_sum      = 0.0
    top_k_max     = 10

    for ex in tqdm(dataset, desc="Retrieval eval"):
        question    = ex["question"]
        gold_titles = set(ex["supporting_facts"]["title"])

        passages = [
            {"title": t, "text": " ".join(s)}
            for t, s in zip(ex["context"]["title"], ex["context"]["sentences"])
        ]

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

    print(f"\n=== Retrieval Metrics ({split}[{start}:{start + max_samples}]) ===")
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
    model_dir = train(
        data_path="data/hard_negatives_train.json",
        output_dir="models/dpr_finetuned",
        max_samples=None,   # local test: set to 50; full run: None
        epochs=3,           # local test: set to 1
        batch_size=4,       # local CPU: 2-4; A100: 16-32
        lr=2e-5,
    )
    # Final evaluation on held-out val slice (train examples 800-999)
    evaluate_recall(model_dir=model_dir, max_samples=200, split="train", start=800)
