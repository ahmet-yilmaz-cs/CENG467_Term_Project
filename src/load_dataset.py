from datasets import load_dataset


def load_hotpotqa(split="validation", max_samples=None, start=0):
    """
    Loads HotpotQA in distractor setting.
    split     : "train" (90k) or "validation" (7.4k)
    max_samples: number of examples to load
    start     : first index to load (for held-out slices)
    """
    print(f"Loading HotpotQA ({split}, start={start})...")
    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)

    if start > 0 or max_samples:
        end = (start + max_samples) if max_samples else len(dataset)
        dataset = dataset.select(range(start, min(end, len(dataset))))

    print(f"  Loaded {len(dataset)} examples")
    print(f"  Columns: {dataset.column_names}")
    return dataset


def inspect_example(dataset, idx=0):
    ex = dataset[idx]
    print("\n--- Example ---")
    print(f"Question : {ex['question']}")
    print(f"Answer   : {ex['answer']}")
    print(f"# Context docs: {len(ex['context']['title'])}")
    for title, sentences in zip(ex['context']['title'], ex['context']['sentences']):
        print(f"  [{title}] {sentences[0][:80]}...")


if __name__ == "__main__":
    ds = load_hotpotqa(split="validation", max_samples=500)
    inspect_example(ds, idx=0)
