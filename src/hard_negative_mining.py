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
    """Her soru için 10 pasajı title + text olarak döner."""
    passages = []
    for title, sentences in zip(
        example["context"]["title"],
        example["context"]["sentences"]
    ):
        passages.append({
            "title": title,
            "text": " ".join(sentences)
        })
    return passages


def get_gold_titles(example):
    """Sorunun cevabını destekleyen gold pasaj başlıklarını döner."""
    return set(example["supporting_facts"]["title"])


def mine_hard_negatives(example, num_hard_negatives=5):
    """
    Bir soru için hard negative mining yapar.

    Dönen yapı:
    {
        "question": ...,
        "answer": ...,
        "positive_passages": [...],   # gold pasajlar
        "hard_negatives": [...]        # yüksek BM25 skoru ama gold değil
    }
    """
    question    = example["question"]
    answer      = example["answer"]
    passages    = build_corpus(example)
    gold_titles = get_gold_titles(example)

    # BM25 ile tüm pasajları sırala
    corpus_tokens = [tokenize(p["text"]) for p in passages]
    bm25          = BM25Okapi(corpus_tokens)
    scores        = bm25.get_scores(tokenize(question))

    # Pasajları skora göre sırala
    ranked = sorted(
        enumerate(passages),
        key=lambda x: scores[x[0]],
        reverse=True
    )

    positive_passages = []
    hard_negatives    = []

    for idx, passage in ranked:
        if passage["title"] in gold_titles:
            positive_passages.append(passage)
        else:
            # Gold olmayan ama yüksek skor → hard negative
            if len(hard_negatives) < num_hard_negatives:
                hard_negatives.append({**passage, "bm25_score": round(float(scores[idx]), 4)})

    # Gold pasaj hiç bulunamadıysa bu örneği atla
    if not positive_passages:
        return None

    return {
        "question":          question,
        "answer":            answer,
        "positive_passages": positive_passages,
        "hard_negatives":    hard_negatives
    }


def run_mining(
    max_samples=1000,
    num_hard_negatives=5,
    output_path="data/hard_negatives_train.json",
    val_ratio=0.2,
    val_output_path="data/hard_negatives_val.json",
):
    """
    Mines hard negatives from the HotpotQA train split and explicitly
    splits the results into a training set and a held-out validation set.

    val_ratio : fraction of mined examples reserved for held-out validation
                (used for checkpoint selection in train_dpr.py, never for
                final ablation evaluation).
    """
    dataset = load_hotpotqa(split="train", max_samples=max_samples)

    results = []
    skipped = 0

    for example in tqdm(dataset, desc="Mining hard negatives"):
        sample = mine_hard_negatives(example, num_hard_negatives=num_hard_negatives)
        if sample is None:
            skipped += 1
            continue
        results.append(sample)

    print(f"\nDone. {len(results)} samples, {skipped} skipped (no gold passage found).")
    print(f"Avg hard negatives per sample: "
          f"{sum(len(r['hard_negatives']) for r in results) / len(results):.2f}")

    # ------------------------------------------------------------------ #
    # Explicit train / validation split at the mining stage.              #
    # Checkpoint selection (train_dpr.py) uses the val portion;           #
    # final ablation evaluation uses the HotpotQA validation split,       #
    # which is completely independent of both sets below.                 #
    # ------------------------------------------------------------------ #
    n_val         = int(len(results) * val_ratio)
    n_train       = len(results) - n_val
    train_results = results[:n_train]
    val_results   = results[n_train:]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(train_results, f, indent=2, ensure_ascii=False)
    print(f"Train set ({len(train_results)} examples) saved to {output_path}")

    with open(val_output_path, "w") as f:
        json.dump(val_results, f, indent=2, ensure_ascii=False)
    print(f"Val   set ({len(val_results)} examples) saved to {val_output_path}")

    return train_results, val_results


if __name__ == "__main__":
    run_mining(
        max_samples=30000,
        num_hard_negatives=5,
        output_path="/content/drive/MyDrive/CENG467/hard_negatives_train_30k.json",
        val_ratio=0.2,
        val_output_path="/content/drive/MyDrive/CENG467/hard_negatives_val_30k.json",
    )
