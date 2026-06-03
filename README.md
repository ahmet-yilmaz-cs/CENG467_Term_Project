# CENG 467 — Advanced RAG with Hard Negative Mining

Term project for CENG 467 Natural Language Understanding and Generation.

İzmir Institute of Technology, Spring 2026.

**Group:** Erman Akıncı · Ahmet Yılmaz · Kaan Cesur

---

## Project Overview

This project implements an Advanced Retrieval-Augmented Generation (RAG) pipeline with Hard Negative Mining on the HotpotQA dataset. The system trains a Dense Passage Retrieval (DPR) model using BM25-mined hard negatives, builds a FAISS vector index for scalable retrieval, and uses Flan-T5-xl as a generative reader to answer multi-hop questions.

**Pipeline:**
1. Hard Negative Mining (BM25) → `data/hard_negatives_train.json` + `data/hard_negatives_val.json`
2. DPR Fine-tuning with contrastive loss (gradient clipping + LR scheduler + curriculum learning)
3. FAISS vector index over validation corpus
4. Flan-T5-xl generative reader on top-3 retrieved passages
5. Ablation study: BM25 vs vanilla DPR vs hard-negative fine-tuned DPR

---

## Repository Structure

```
CENG467_Term_Project/
├── src/
│   ├── load_dataset.py          # HotpotQA loading utility
│   ├── baseline_bm25.py         # BM25 retrieval baseline
│   ├── baseline_dpr.py          # Vanilla DPR retrieval baseline
│   ├── hard_negative_mining.py  # BM25-based hard negative generation (train/val split)
│   ├── train_dpr.py             # DPR fine-tuning with contrastive loss
│   ├── reader.py                # FAISS retrieval + Flan-T5-xl reader pipeline
│   └── ablation.py              # Unified ablation study (BM25 / vanilla / DPR+HN)
├── data/
│   ├── hard_negatives_train.json    # Training hard negatives (80%)
│   └── hard_negatives_val.json      # Held-out validation hard negatives (20%)
├── results/                     # Evaluation outputs (generated at runtime)
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone -b feature/reader https://github.com/ahmet-yilmaz-cs/CENG467_Term_Project.git
cd CENG467_Term_Project
pip install -r requirements.txt
```

> **Note:** Fine-tuned model weights are not included in the repository due to size (each epoch checkpoint is ~870 MB). Either train from scratch (see steps below) or download from the shared Google Drive link and place under `models/`.

Expected model structure after training (one directory per epoch):
```
models/
└── dpr_hn_v2/
    ├── epoch1/
    │   ├── question_encoder/
    │   └── context_encoder/
    ├── epoch2/
    │   ├── question_encoder/
    │   └── context_encoder/
    └── epoch3/          # best on global retrieval — used in the ablation
        ├── question_encoder/
        └── context_encoder/
```

---

## Reproducing Results

### Step 1 — BM25 Baseline
```bash
python src/baseline_bm25.py
# Output: results/bm25_results.json
```

### Step 2 — Vanilla DPR Baseline
```bash
python src/baseline_dpr.py
# Output: results/dpr_results.json
# Downloads ~800 MB model weights automatically
```

### Step 3 — Hard Negative Mining

**5k training examples:**
```bash
python src/hard_negative_mining.py
# Output: data/hard_negatives_train.json  (800 examples, 80%)
#         data/hard_negatives_val.json    (200 examples, 20% held-out)
```

**30k training examples:**
```python
from src.hard_negative_mining import run_mining
run_mining(
    max_samples=30000,
    output_path="data/hard_negatives_30k_train.json",
    val_output_path="data/hard_negatives_30k_val.json"
)
```

### Step 4 — DPR Fine-tuning (GPU required)

```python
from src.train_dpr import train
train(
    data_path="data/hard_negatives_train.json",
    output_dir="models/dpr_hn_v2",
    epochs=3, batch_size=16, lr=2e-5
)
# Output: models/dpr_hn_v2/epoch1/, epoch2/, epoch3/
```

Training features:
- **Curriculum learning**: 1 → 3 → 5 hard negatives per epoch
- **Gradient clipping**: max_norm=1.0
- **LR scheduler**: linear warmup (10%) + linear decay
- **Per-epoch checkpoints**: each epoch is saved to its own directory; the
  final checkpoint is chosen by **global retrieval** (Step 6), *not* by the
  mini-pool recall during training — that pool (~7 candidates) is near-saturated
  and unreliable for selection.
- Recommended: Google Colab with A100 GPU

### Step 5 — Reader Pipeline (GPU recommended)
```bash
python src/reader.py
# Output: results/reader_results.json
```

### Step 6 — Ablation Study (GPU required)
```bash
python src/ablation.py
# Output: results/ablation_results.json
```

---

## Results

Final ablation on the HotpotQA validation split (n=500, retrieval over the
pooled passages of the eval set, Flan-T5-xl reader on top-3 passages):

| Config | R@1 | R@3 | R@5 | F1 | EM |
|---|---|---|---|---|---|
| BM25 | 0.704 | 0.862 | 0.904 | 0.460 | 0.352 |
| DPR (vanilla) | 0.714 | 0.854 | 0.888 | 0.442 | 0.346 |
| **DPR + HN** | **0.844** | **0.936** | **0.962** | **0.539** | **0.428** |

Hard-negative fine-tuning improves retrieval R@1 by **+13 points** over vanilla
DPR (and +14 over BM25), and the gain propagates end-to-end (**+0.10 F1 / +0.08
EM**). Vanilla (NQ-pretrained) DPR only matches the lexical BM25 baseline — the
improvement comes from the hard-negative fine-tuning.

---

## Dataset

- **HotpotQA** (distractor setting) — Yang et al., 2018
- Loaded via HuggingFace `datasets` library
- Training: `train` split (hard negative mining)
- Evaluation: `validation` split (500 examples)

---

## References

- Karpukhin et al. (2020) — Dense Passage Retrieval
- Lewis et al. (2020) — Retrieval-Augmented Generation
- Yang et al. (2018) — HotpotQA
- Chung et al. (2022) — Flan-T5
