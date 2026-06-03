import json
import os
import string
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from load_dataset import load_hotpotqa


def tokenize(text):
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return text.split()


def build_corpus(example):
    """
    HotpotQA distractor setting: her soru için 10 paragraf var (2 gold + 8 distractor).
    Bunları birleştirip BM25 corpus'u oluşturuyoruz.
    """
    titles = example["context"]["title"]
    sentences_list = example["context"]["sentences"]
    passages = []
    for title, sentences in zip(titles, sentences_list):
        text = " ".join(sentences)
        passages.append({"title": title, "text": text})
    return passages


def retrieve_bm25(question, passages, top_k=5):
    corpus_tokens = [tokenize(p["text"]) for p in passages]
    bm25 = BM25Okapi(corpus_tokens)
    query_tokens = tokenize(question)
    scores = bm25.get_scores(query_tokens)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return ranked_indices[:top_k]


def normalize(text):
    return text.lower().translate(str.maketrans("", "", string.punctuation)).strip()


def token_f1(prediction, ground_truth):
    pred_tokens = normalize(prediction).split()
    gt_tokens = normalize(ground_truth).split()
    common = set(pred_tokens) & set(gt_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction, ground_truth):
    return int(normalize(prediction) == normalize(ground_truth))


def run_bm25_baseline(max_samples=500, top_k=5):
    dataset = load_hotpotqa(split="validation", max_samples=max_samples)

    recall_at_k = 0
    f1_total = 0
    em_total = 0
    results = []

    for ex in tqdm(dataset, desc="BM25 Retrieval"):
        question = ex["question"]
        answer = ex["answer"]
        passages = build_corpus(ex)

        # Altın (gold) başlıklar: hangi pasajların doğru olduğu
        gold_titles = set(ex["supporting_facts"]["title"])

        # BM25 ile top-k pasaj al
        top_indices = retrieve_bm25(question, passages, top_k=top_k)
        retrieved_titles = {passages[i]["title"] for i in top_indices}

        # Recall@K: gold pasajlardan en az biri alındı mı?
        hit = int(bool(gold_titles & retrieved_titles))
        recall_at_k += hit

        # Cevap tahmini: en iyi pasajın ilk cümlesi (basit yaklaşım)
        best_passage_text = passages[top_indices[0]]["text"]
        predicted_answer = best_passage_text.split(".")[0]

        f1 = token_f1(predicted_answer, answer)
        em = exact_match(predicted_answer, answer)
        f1_total += f1
        em_total += em

        results.append({
            "question": question,
            "gold_answer": answer,
            "predicted": predicted_answer,
            "hit": hit,
            "f1": f1,
            "em": em,
        })

    n = len(dataset)
    metrics = {
        "model": "BM25",
        "num_samples": n,
        "top_k": top_k,
        f"Recall@{top_k}": round(recall_at_k / n, 4),
        "F1": round(f1_total / n, 4),
        "Exact Match": round(em_total / n, 4),
    }

    print("\n=== BM25 Baseline Results ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    os.makedirs("results", exist_ok=True)
    with open("results/bm25_results.json", "w") as f:
        json.dump({"metrics": metrics, "predictions": results[:50]}, f, indent=2)

    print("\nResults saved to results/bm25_results.json")
    return metrics


if __name__ == "__main__":
    run_bm25_baseline(max_samples=500, top_k=5)
