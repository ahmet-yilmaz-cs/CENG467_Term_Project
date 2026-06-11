"""
Reader comparison on a fixed retriever (DPR + HN).

Compares an encoder-decoder reader (Flan-T5-xl) with two decoder-only
instruction models (Qwen2.5-7B-Instruct, Phi-4-mini-Instruct). The decoder-only
models are 4-bit quantised and use a few-shot prompt with answer
post-processing; Flan-T5-xl uses its simpler native prompt (the multi-part
instruction degrades it). Retrieval is computed once and cached, then each
reader is evaluated in turn (loaded, scored, freed) to fit on a single GPU.

Finding: the 3B encoder-decoder slightly outperforms the larger decoder-only
models, and the reader input-token budget (max_length) is the single largest
end-to-end lever.

Shared retrieval/metric helpers are imported from ablation.py.
Run from the src/ directory:  python reader_comparison.py
Requires: bitsandbytes (4-bit quantisation), accelerate.
"""
import json
import os
import re

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
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

# Few-shot + concise-answer instruction. Helps decoder-only instruction models;
# the post-processing below strips any leftover prefixes.
FEWSHOT_PROMPT = (
    "You answer questions from the given context. Reply with ONLY the short "
    "answer phrase — no explanation. For yes/no questions reply exactly 'yes' "
    "or 'no'.\n\n"
    "Context: The Eiffel Tower is in Paris. The Louvre is in Paris.\n"
    "Question: Are the Eiffel Tower and the Louvre in the same city?\nAnswer: yes\n\n"
    "Context: Inception is a 2010 film directed by Christopher Nolan.\n"
    "Question: Who directed Inception?\nAnswer: Christopher Nolan\n\n"
    "Context: {ctx}\nQuestion: {q}\nAnswer:"
)

# Flan-T5 follows a simple native prompt better than the multi-part instruction.
NATIVE_PROMPT = (
    "Answer the question based on the context below.\n"
    "Context: {ctx}\nQuestion: {q}\nAnswer:"
)

READERS = [
    {"name": "flan-t5-xl", "id": "google/flan-t5-xl",
     "type": "seq2seq", "quant": False, "prompt": "native"},
    {"name": "Qwen2.5-7B", "id": "Qwen/Qwen2.5-7B-Instruct",
     "type": "causal", "quant": True, "prompt": "fewshot"},
    {"name": "Phi-4-mini", "id": "microsoft/Phi-4-mini-instruct",
     "type": "causal", "quant": True, "prompt": "fewshot"},
]


# ---------------------------------------------------------------------------
# Reader helpers
# ---------------------------------------------------------------------------

def post_process(answer):
    """Keep the first line and strip common answer prefixes / punctuation."""
    answer = answer.strip().split("\n")[0].strip()
    answer = re.sub(
        r"^(the answer is|answer|based on the context[,:]?)\s*[:\-]?\s*",
        "", answer, flags=re.I,
    )
    return answer.strip(" .,\"'")


def load_reader(spec):
    tokenizer = AutoTokenizer.from_pretrained(spec["id"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_cls = AutoModelForSeq2SeqLM if spec["type"] == "seq2seq" else AutoModelForCausalLM
    if spec["quant"]:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        model = model_cls.from_pretrained(spec["id"], quantization_config=bnb, device_map="auto")
    else:
        model = model_cls.from_pretrained(spec["id"]).to(device)
    model.eval()
    return tokenizer, model


def answer_question(spec, tokenizer, model, question, context, max_length=1024):
    template = NATIVE_PROMPT if spec["prompt"] == "native" else FEWSHOT_PROMPT
    prompt = template.format(ctx=context, q=question)

    if spec["type"] == "seq2seq":
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=max_length).to(device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=32, do_sample=False)
        return post_process(tokenizer.decode(output[0], skip_special_tokens=True))

    # Decoder-only instruction model: chat template, then strip the prompt tokens.
    enc = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        return_tensors="pt", add_generation_prompt=True, return_dict=True,
    ).to(device)
    with torch.no_grad():
        output = model.generate(**enc, max_new_tokens=32, do_sample=False,
                                pad_token_id=tokenizer.eos_token_id)
    generated = output[0][enc["input_ids"].shape[1]:]
    return post_process(tokenizer.decode(generated, skip_special_tokens=True))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_reader_comparison(
    model_dir="models/dpr_hn_v2/epoch3",
    max_samples=500,
    top_k=3,
    output_path="results/reader_comparison.json",
):
    os.makedirs("results", exist_ok=True)
    print(f"Device: {device}")

    dataset  = load_hotpotqa(split="validation", max_samples=max_samples)
    passages = collect_passages(dataset)
    print(f"Corpus: {len(passages)} unique passages")

    # Fixed retriever (DPR + HN): retrieve top-k once and cache the context.
    q_tok, q_enc, c_tok, c_enc = load_dpr_encoders(model_dir)
    faiss_idx = build_faiss_index(passages, c_tok, c_enc)
    cache = []
    for ex in tqdm(dataset, desc="Retrieving (DPR + HN)"):
        top = retrieve_dpr(ex["question"], faiss_idx, passages, q_tok, q_enc, top_k)
        context = " ".join(p["title"] + " " + p["text"] for p in top)[:2000]
        cache.append({"q": ex["question"], "a": ex["answer"], "ctx": context})
    del q_enc, c_enc, faiss_idx
    if device == "cuda":
        torch.cuda.empty_cache()

    # Evaluate each reader in turn (one model in memory at a time).
    rows = []
    for spec in READERS:
        print(f"\n[{spec['name']}] loading...")
        tokenizer, model = load_reader(spec)
        f1_total = em_total = 0.0
        for c in tqdm(cache, desc=spec["name"]):
            pred = answer_question(spec, tokenizer, model, c["q"], c["ctx"])
            f1_total += token_f1(pred, c["a"])
            em_total += exact_match(pred, c["a"])
        n = len(cache)
        rows.append({"reader": spec["name"],
                     "F1": round(f1_total / n, 4),
                     "EM": round(em_total / n, 4)})
        del model, tokenizer
        if device == "cuda":
            torch.cuda.empty_cache()

    print("\n=== Reader comparison (DPR + HN retriever, top-3 passages) ===")
    print(f"{'Reader':<14}{'F1':>9}{'EM':>9}")
    print("-" * 32)
    for r in rows:
        print(f"{r['reader']:<14}{r['F1']:>9.4f}{r['EM']:>9.4f}")

    with open(output_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved to {output_path}")
    return rows


if __name__ == "__main__":
    run_reader_comparison(
        model_dir="models/dpr_hn_v2/epoch3",
        max_samples=500,
        top_k=3,
        output_path="results/reader_comparison.json",
    )
