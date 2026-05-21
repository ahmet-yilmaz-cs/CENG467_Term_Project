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
)

from load_dataset import load_hotpotqa

QUESTION_MODEL = "facebook/dpr-question_encoder-single-nq-base"
CONTEXT_MODEL = "facebook/dpr-ctx_encoder-single-nq-base"

device = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_hard_negatives(path="data/hard_negatives.json", max_samples=None):
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
    data_path="data/hard_negatives.json",
    output_dir="models/dpr_finetuned",
    max_samples=None,
    epochs=2,
    batch_size=4,
    lr=2e-5,
    recall_eval_samples=500,
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

    best_recall = 0.0

    for epoch in range(epochs):
        total_loss = 0.0
        steps = 0

        for start in tqdm(range(0, len(data), batch_size), desc=f"Epoch {epoch + 1}/{epochs}"):
            batch = data[start : start + batch_size]
            actual_bs = len(batch)

            questions = [ex["question"] for ex in batch]

            # First positive passage per question
            positives = [
                ex["positive_passages"][0]["title"] + " " + ex["positive_passages"][0]["text"]
                for ex in batch
            ]

            # All hard negatives concatenated (variable count per question)
            hard_negs = [
                p["title"] + " " + p["text"]
                for ex in batch
                for p in ex["hard_negatives"]
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
            optimizer.step()

            total_loss += loss.item()
            steps += 1

        avg_loss = total_loss / steps
        print(f"  Epoch {epoch + 1} — avg loss: {avg_loss:.4f}")

        # Save current epoch to output_dir, evaluate Recall@5, keep best
        q_encoder.save_pretrained(f"{output_dir}/question_encoder")
        q_tokenizer.save_pretrained(f"{output_dir}/question_encoder")
        c_encoder.save_pretrained(f"{output_dir}/context_encoder")
        c_tokenizer.save_pretrained(f"{output_dir}/context_encoder")

        recall = evaluate_recall(model_dir=output_dir, max_samples=recall_eval_samples, top_k=5)

        if recall > best_recall:
            best_recall = recall
            q_encoder.save_pretrained(f"{best_dir}/question_encoder")
            q_tokenizer.save_pretrained(f"{best_dir}/question_encoder")
            c_encoder.save_pretrained(f"{best_dir}/context_encoder")
            c_tokenizer.save_pretrained(f"{best_dir}/context_encoder")
            print(f"  ✓ Best model saved (Recall@5: {best_recall:.4f})")

    print(f"\nTraining complete. Best Recall@5: {best_recall:.4f}")
    print(f"Best model saved to {best_dir}/")
    return best_dir


# ---------------------------------------------------------------------------
# Evaluation (Recall@5) using fine-tuned model
# ---------------------------------------------------------------------------

def evaluate_recall(model_dir="models/dpr_finetuned", max_samples=500, top_k=5):
    print(f"\nEvaluating fine-tuned model from {model_dir}...")

    q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(f"{model_dir}/question_encoder")
    q_encoder   = DPRQuestionEncoder.from_pretrained(f"{model_dir}/question_encoder").to(device).eval()
    c_tokenizer = DPRContextEncoderTokenizer.from_pretrained(f"{model_dir}/context_encoder")
    c_encoder   = DPRContextEncoder.from_pretrained(f"{model_dir}/context_encoder").to(device).eval()

    dataset = load_hotpotqa(split="validation", max_samples=max_samples)
    hits = 0

    for ex in tqdm(dataset, desc="Recall@5 eval"):
        question    = ex["question"]
        gold_titles = set(ex["supporting_facts"]["title"])

        titles    = ex["context"]["title"]
        sentences = ex["context"]["sentences"]
        passages  = [
            {"title": t, "text": " ".join(s)}
            for t, s in zip(titles, sentences)
        ]

        # Encode question
        q_inputs = q_tokenizer(
            question, return_tensors="pt", truncation=True, max_length=64,
        ).to(device)
        with torch.no_grad():
            q_emb = q_encoder(**q_inputs).pooler_output.squeeze(0).cpu().numpy()

        # Encode passages
        p_texts = [p["title"] + " " + p["text"] for p in passages]
        p_inputs = c_tokenizer(
            p_texts, return_tensors="pt",
            truncation=True, max_length=256, padding=True,
        ).to(device)
        with torch.no_grad():
            p_embs = c_encoder(**p_inputs).pooler_output.cpu().numpy()

        scores         = p_embs @ q_emb
        top_indices    = np.argsort(scores)[::-1][:top_k]
        retrieved      = {passages[i]["title"] for i in top_indices}
        hits          += int(bool(gold_titles & retrieved))

    recall = hits / len(dataset)
    print(f"\n=== Fine-tuned DPR Results ===")
    print(f"  Recall@{top_k}: {recall:.4f}  ({hits}/{len(dataset)})")
    print(f"  (Vanilla DPR baseline: 0.982)")

    os.makedirs("results", exist_ok=True)
    output = {
        "model": "DPR + Hard Negatives",
        "model_dir": model_dir,
        "num_samples": len(dataset),
        "top_k": top_k,
        f"Recall@{top_k}": round(recall, 4),
    }
    out_path = "results/dpr_hn_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Results saved to {out_path}")
    return recall


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model_dir = train(
        data_path="data/hard_negatives.json",
        output_dir="models/dpr_finetuned",
        max_samples=None,   # local test: set to 50; full run: None
        epochs=3,           # local test: set to 1
        batch_size=4,       # local CPU: 2-4; Colab T4 GPU: use 8 when calling train() directly
        lr=2e-5,
    )
    evaluate_recall(model_dir=model_dir, max_samples=500, top_k=5)
