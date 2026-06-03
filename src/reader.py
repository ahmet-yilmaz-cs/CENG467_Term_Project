import faiss
import json
import os
import string

import numpy as np
import torch
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
# QA model — generative reader (Flan-T5-xl)
# ---------------------------------------------------------------------------

QA_MODEL = "google/flan-t5-xl"

# DPR fallback (vanilla) if fine-tuned model is not available
VANILLA_Q_MODEL = "facebook/dpr-question_encoder-single-nq-base"
VANILLA_C_MODEL = "facebook/dpr-ctx_encoder-single-nq-base"

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


def collect_passages(dataset):
    """Collect all unique passages (deduplicated by title) across the dataset."""
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
# DPR encoders
# ---------------------------------------------------------------------------

def load_dpr_encoders(model_dir=None):
    """
    model_dir: path to fine-tuned model (e.g. 'models/dpr_hn_v2/epoch3').
               If None, uses vanilla DPR. If a path is given it MUST exist
               (fails loudly instead of silently falling back to vanilla).
    """
    if model_dir:
        q_enc_path = f"{model_dir}/question_encoder"
        c_enc_path = f"{model_dir}/context_encoder"
        assert os.path.isdir(q_enc_path), f"Fine-tuned model not found: {q_enc_path}"
        print(f"Loading fine-tuned DPR from {model_dir}/")
    else:
        print("Using vanilla DPR.")
        q_enc_path = VANILLA_Q_MODEL
        c_enc_path = VANILLA_C_MODEL

    q_tokenizer = DPRQuestionEncoderTokenizerFast.from_pretrained(q_enc_path)
    q_encoder   = DPRQuestionEncoder.from_pretrained(q_enc_path).to(device).eval()
    c_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained(c_enc_path)
    c_encoder   = DPRContextEncoder.from_pretrained(c_enc_path).to(device).eval()
    return q_tokenizer, q_encoder, c_tokenizer, c_encoder


# ---------------------------------------------------------------------------
# FAISS index
# ---------------------------------------------------------------------------

def build_faiss_index(passages, c_tokenizer, c_encoder, batch_size=32):
    """
    Encode all passages with the context encoder and build a FAISS
    IndexFlatIP (exact inner-product / dot-product search).

    Returns the faiss.Index object. Passage order is preserved so
    index position i corresponds to passages[i].
    """
    print(f"Building FAISS index for {len(passages)} passages...")
    all_embs = []

    for i in tqdm(range(0, len(passages), batch_size), desc="Encoding passages"):
        batch_texts = [
            p["title"] + " " + p["text"]
            for p in passages[i : i + batch_size]
        ]
        inputs = c_tokenizer(
            batch_texts, return_tensors="pt",
            truncation=True, max_length=256, padding=True,
        ).to(device)
        with torch.no_grad():
            embs = c_encoder(**inputs).pooler_output.cpu().numpy()
        all_embs.append(embs)

    matrix = np.vstack(all_embs).astype("float32")
    dim    = matrix.shape[1]           # 768 for DPR
    index  = faiss.IndexFlatIP(dim)    # exact inner-product search
    index.add(matrix)
    print(f"  Index built: {index.ntotal} vectors, dim={dim}")
    return index


# ---------------------------------------------------------------------------
# Retrieval via FAISS
# ---------------------------------------------------------------------------

def retrieve(question, faiss_index, passages, q_tokenizer, q_encoder, top_k=3):
    """
    Encode the question, search the global FAISS index, and return the
    top_k passage dicts ranked by inner-product score.
    """
    q_inputs = q_tokenizer(
        question, return_tensors="pt", truncation=True, max_length=64,
    ).to(device)
    with torch.no_grad():
        q_emb = q_encoder(**q_inputs).pooler_output.cpu().numpy().astype("float32")

    _scores, indices = faiss_index.search(q_emb, top_k)   # (1, top_k)
    return [passages[i] for i in indices[0]]


# ---------------------------------------------------------------------------
# Reader: extractive QA over retrieved passages
# ---------------------------------------------------------------------------

def read(question, top_passages, reader_model, reader_tokenizer):
    """
    Run Flan-T5-xl on the concatenated top passages and return the generated answer.
    Uses model.generate() directly — the "text2text-generation" pipeline task
    was removed in recent transformers versions.
    """
    context = " ".join([p["title"] + " " + p["text"] for p in top_passages])
    # Truncate context to avoid exceeding model input limits
    context = context[:2000]
    prompt  = (
        f"Answer the question based on the context below.\n"
        f"Context: {context}\n"
        f"Question: {question}\n"
        f"Answer:"
    )
    inputs = reader_tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=512,
    ).to(device)
    with torch.no_grad():
        output = reader_model.generate(**inputs, max_new_tokens=50, do_sample=False)
    answer = reader_tokenizer.decode(output[0], skip_special_tokens=True).strip()
    return answer, 1.0  # score placeholder for API compatibility


# ---------------------------------------------------------------------------
# Full pipeline: build index → retrieve → read → evaluate
# ---------------------------------------------------------------------------

def run_reader_pipeline(
    model_dir="models/dpr_hn_v2/epoch3",
    max_samples=500,
    top_k=3,
    output_path="results/reader_results.json",
):
    print(f"Device: {device}")
    os.makedirs("results", exist_ok=True)

    # Load DPR encoders
    q_tokenizer, q_encoder, c_tokenizer, c_encoder = load_dpr_encoders(model_dir)

    # Load QA reader (model.generate directly; no pipeline)
    print(f"Loading QA reader ({QA_MODEL})...")
    reader_tokenizer = AutoTokenizer.from_pretrained(QA_MODEL)
    reader_model     = AutoModelForSeq2SeqLM.from_pretrained(QA_MODEL).to(device).eval()
    print("  Reader loaded.")

    dataset = load_hotpotqa(split="validation", max_samples=max_samples)

    # ------------------------------------------------------------------
    # Build a single global FAISS index over all passages in the eval set.
    # This simulates a real vector-database retrieval scenario where the
    # retriever searches a shared corpus rather than a per-question pool.
    # ------------------------------------------------------------------
    all_passages = collect_passages(dataset)
    faiss_index  = build_faiss_index(all_passages, c_tokenizer, c_encoder)

    recall_at_k = 0
    f1_total    = 0
    em_total    = 0
    results     = []

    for ex in tqdm(dataset, desc="Reader pipeline"):
        question    = ex["question"]
        answer      = ex["answer"]
        gold_titles = set(ex["supporting_facts"]["title"])

        # Step 1: retrieve top-k passages from the global FAISS index
        top_passages     = retrieve(question, faiss_index, all_passages,
                                    q_tokenizer, q_encoder, top_k=top_k)
        retrieved_titles = {p["title"] for p in top_passages}
        hit              = int(bool(gold_titles & retrieved_titles))
        recall_at_k     += hit

        # Step 2: extract answer from retrieved passages
        predicted, score = read(question, top_passages, reader_model, reader_tokenizer)

        f1 = token_f1(predicted, answer)
        em = exact_match(predicted, answer)
        f1_total += f1
        em_total += em

        results.append({
            "question":         question,
            "gold_answer":      answer,
            "predicted":        predicted,
            "reader_score":     round(score, 4),
            "retrieved_titles": list(retrieved_titles),
            "hit":              hit,
            "f1":               round(f1, 4),
            "em":               em,
        })

    n = len(dataset)
    metrics = {
        "model":             "DPR + FAISS + Reader (flan-t5-xl)",
        "retriever":         model_dir if os.path.isdir(str(model_dir)) else "vanilla DPR",
        "num_samples":       n,
        "corpus_size":       len(all_passages),
        "top_k":             top_k,
        f"Recall@{top_k}":   round(recall_at_k / n, 4),
        "F1":                round(f1_total / n, 4),
        "Exact Match":       round(em_total / n, 4),
    }

    print("\n=== Reader Pipeline Results ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    with open(output_path, "w") as f:
        json.dump({"metrics": metrics, "predictions": results[:50]}, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")
    return metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_reader_pipeline(
        model_dir="models/dpr_hn_v2/epoch3",
        max_samples=500,
        top_k=3,
        output_path="results/reader_results.json",
    )
