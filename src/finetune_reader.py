"""
Reader fine-tuning (LoRA) — a SEPARATE experiment, not part of the main ablation.

Fine-tunes Flan-T5-xl on the HotpotQA *train* split (gold-passage context ->
answer) with LoRA, then evaluates on the validation split using the DPR + HN
retriever. Kept separate so the main ablation's fixed zero-shot reader cleanly
isolates the retrieval contribution; this reports the best achievable
end-to-end result.

Trains on the TRAIN split only (no validation leakage). Gold supporting
passages are placed first so the answer's evidence survives truncation.

Result (8k train examples, 1 epoch, LoRA r=16): the gain is modest
(F1 0.583 -> 0.613, EM 0.466 -> 0.492), confirming the reader is not the
bottleneck — retrieval is.

Shared retrieval/metric helpers are imported from ablation.py.
Run from the src/ directory:  python finetune_reader.py
Requires: peft, bitsandbytes, accelerate.
"""
import os

import torch
from tqdm import tqdm

from load_dataset import load_hotpotqa
from ablation import (
    token_f1,
    exact_match,
    collect_passages,
    load_dpr_encoders,
    build_faiss_index,
    retrieve_dpr,
    device,
)
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig

QA_MODEL = "google/flan-t5-xl"
PROMPT = (
    "Answer the question based on the context below.\n"
    "Context: {ctx}\nQuestion: {q}\nAnswer:"
)


def _gold_first_context(example, max_chars=2000):
    """Concatenate the example's passages with gold (supporting) passages first."""
    gold = set(example["supporting_facts"]["title"])
    passages = sorted(
        zip(example["context"]["title"], example["context"]["sentences"]),
        key=lambda tp: tp[0] not in gold,   # gold titles first (False sorts before True)
    )
    return " ".join(t + " " + " ".join(s) for t, s in passages)[:max_chars]


# ---------------------------------------------------------------------------
# Fine-tuning (LoRA on the train split)
# ---------------------------------------------------------------------------

def finetune(adapter_dir="models/reader_lora", n_train=8000, epochs=1,
             batch_size=8, lr=2e-4, max_len=1024):
    from peft import (LoraConfig, TaskType, get_peft_model,
                      prepare_model_for_kbit_training)

    tokenizer = AutoTokenizer.from_pretrained(QA_MODEL)
    base = AutoModelForSeq2SeqLM.from_pretrained(
        QA_MODEL,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map="auto",
    )
    base = prepare_model_for_kbit_training(base)
    model = get_peft_model(base, LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM, r=16, lora_alpha=32,
        lora_dropout=0.05, target_modules=["q", "v"]))
    model.print_trainable_parameters()

    data = load_hotpotqa(split="train", max_samples=n_train)
    pairs = [(PROMPT.format(ctx=_gold_first_context(ex), q=ex["question"]), ex["answer"])
             for ex in data]
    print(f"{len(pairs)} training pairs")

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    model.train()
    for epoch in range(epochs):
        total, steps = 0.0, 0
        for i in tqdm(range(0, len(pairs), batch_size), desc=f"Epoch {epoch + 1}/{epochs}"):
            batch = pairs[i:i + batch_size]
            enc = tokenizer([x[0] for x in batch], return_tensors="pt", padding=True,
                            truncation=True, max_length=max_len).to(device)
            labels = tokenizer(text_target=[x[1] for x in batch], return_tensors="pt",
                               padding=True, truncation=True, max_length=32).input_ids
            labels[labels == tokenizer.pad_token_id] = -100
            out = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                        labels=labels.to(device))
            out.loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total += out.loss.item()
            steps += 1
        print(f"  avg loss: {total / steps:.4f}")

    os.makedirs(adapter_dir, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"Saved LoRA adapter to {adapter_dir}/")
    return adapter_dir


# ---------------------------------------------------------------------------
# Evaluation (validation split, DPR + HN retriever)
# ---------------------------------------------------------------------------

def evaluate(adapter_dir="models/reader_lora", model_dir="models/dpr_hn_v2/epoch3",
             max_samples=500, top_k=3, max_len=1024):
    from peft import PeftModel

    dataset  = load_hotpotqa(split="validation", max_samples=max_samples)
    passages = collect_passages(dataset)
    q_tok, q_enc, c_tok, c_enc = load_dpr_encoders(model_dir)
    faiss_idx = build_faiss_index(passages, c_tok, c_enc)

    reader_tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    base = AutoModelForSeq2SeqLM.from_pretrained(QA_MODEL).to(device).eval()
    reader = PeftModel.from_pretrained(base, adapter_dir).eval()

    f1_total = em_total = 0.0
    for ex in tqdm(dataset, desc="Eval (fine-tuned reader)"):
        top = retrieve_dpr(ex["question"], faiss_idx, passages, q_tok, q_enc, top_k)
        context = " ".join(p["title"] + " " + p["text"] for p in top)[:2000]
        prompt = PROMPT.format(ctx=context, q=ex["question"])
        inputs = reader_tokenizer(prompt, return_tensors="pt", truncation=True,
                                  max_length=max_len).to(device)
        with torch.no_grad():
            output = reader.generate(**inputs, max_new_tokens=32, do_sample=False)
        pred = reader_tokenizer.decode(output[0], skip_special_tokens=True).strip()
        f1_total += token_f1(pred, ex["answer"])
        em_total += exact_match(pred, ex["answer"])

    n = len(dataset)
    metrics = {"F1": round(f1_total / n, 4), "EM": round(em_total / n, 4)}
    print(f"\nDPR + HN + fine-tuned reader: F1 {metrics['F1']}  EM {metrics['EM']}")
    return metrics


if __name__ == "__main__":
    finetune()
    evaluate()
