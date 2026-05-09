import json
import string
import numpy as np
from tqdm import tqdm
import torch
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
from load_dataset import load_hotpotqa


QUESTION_MODEL = "facebook/dpr-question_encoder-single-nq-base"
CONTEXT_MODEL  = "facebook/dpr-ctx_encoder-single-nq-base"

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_encoders():
    print("Loading DPR encoders...")
    q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(QUESTION_MODEL)
    q_encoder   = DPRQuestionEncoder.from_pretrained(QUESTION_MODEL).to(device).eval()
    c_tokenizer = DPRContextEncoderTokenizer.from_pretrained(CONTEXT_MODEL)
    c_encoder   = DPRContextEncoder.from_pretrained(CONTEXT_MODEL).to(device).eval()
    print("  Encoders loaded.")
    return q_tokenizer, q_encoder, c_tokenizer, c_encoder


def encode_question(question, q_tokenizer, q_encoder):
    inputs = q_tokenizer(question, return_tensors="pt", truncation=True, max_length=64).to(device)
    with torch.no_grad():
        embedding = q_encoder(**inputs).pooler_output  # (1, 768)
    return embedding.squeeze(0).cpu().numpy()


def encode_passages(passages, c_tokenizer, c_encoder, batch_size=16):
    all_embeddings = []
    for i in range(0, len(passages), batch_size):
        batch_texts = [p["title"] + " " + p["text"] for p in passages[i:i+batch_size]]
        inputs = c_tokenizer(
            batch_texts, return_tensors="pt",
            truncation=True, max_length=256, padding=True
        ).to(device)
        with torch.no_grad():
            embeddings = c_encoder(**inputs).pooler_output  # (batch, 768)
        all_embeddings.append(embeddings.cpu().numpy())
    return np.vstack(all_embeddings)  # (num_passages, 768)


def retrieve_dpr(q_embedding, passage_embeddings, top_k=5):
    # Dot product similarity
    scores = passage_embeddings @ q_embedding  # (num_passages,)
    ranked_indices = np.argsort(scores)[::-1][:top_k]
    return ranked_indices.tolist()


def build_corpus(example):
    titles = example["context"]["title"]
    sentences_list = example["context"]["sentences"]
    passages = []
    for title, sentences in zip(titles, sentences_list):
        text = " ".join(sentences)
        passages.append({"title": title, "text": text})
    return passages


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


def run_dpr_baseline(max_samples=500, top_k=5):
    dataset = load_hotpotqa(split="validation", max_samples=max_samples)
    q_tokenizer, q_encoder, c_tokenizer, c_encoder = load_encoders()

    recall_at_k = 0
    f1_total    = 0
    em_total    = 0
    results     = []

    for ex in tqdm(dataset, desc="DPR Retrieval"):
        question    = ex["question"]
        answer      = ex["answer"]
        passages    = build_corpus(ex)
        gold_titles = set(ex["supporting_facts"]["title"])

        q_emb  = encode_question(question, q_tokenizer, q_encoder)
        p_embs = encode_passages(passages, c_tokenizer, c_encoder)

        top_indices     = retrieve_dpr(q_emb, p_embs, top_k=top_k)
        retrieved_titles = {passages[i]["title"] for i in top_indices}

        hit = int(bool(gold_titles & retrieved_titles))
        recall_at_k += hit

        best_passage_text = passages[top_indices[0]]["text"]
        predicted_answer  = best_passage_text.split(".")[0]

        f1 = token_f1(predicted_answer, answer)
        em = exact_match(predicted_answer, answer)
        f1_total += f1
        em_total += em

        results.append({
            "question":    question,
            "gold_answer": answer,
            "predicted":   predicted_answer,
            "hit":         hit,
            "f1":          f1,
            "em":          em,
        })

    n = len(dataset)
    metrics = {
        "model":          "DPR (vanilla)",
        "num_samples":    n,
        "top_k":          top_k,
        f"Recall@{top_k}": round(recall_at_k / n, 4),
        "F1":             round(f1_total / n, 4),
        "Exact Match":    round(em_total / n, 4),
    }

    print("\n=== DPR Baseline Results ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    with open("results/dpr_results.json", "w") as f:
        json.dump({"metrics": metrics, "predictions": results[:50]}, f, indent=2)

    print("\nResults saved to results/dpr_results.json")
    return metrics


if __name__ == "__main__":
    run_dpr_baseline(max_samples=500, top_k=5)
