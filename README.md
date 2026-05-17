# CENG 467 — Advanced RAG with Hard Negative Mining

Term project for CENG 467 Natural Language Understanding and Generation.

İzmir Institute of Technology, Spring 2026.

**Group:** Erman Akıncı · Ahmet Yılmaz · Kaan Cesur

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
# BM25 baseline (downloads dataset automatically)
python src/baseline_bm25.py

# DPR baseline (downloads ~800 MB model weights)
python src/baseline_dpr.py

# Hard negative mining
python src/hard_negative_mining.py

# DPR fine-tuning with hard negatives
python src/train_dpr.py
```