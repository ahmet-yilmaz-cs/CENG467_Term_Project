from datasets import load_dataset


def load_hotpotqa(split="validation", max_samples=None):
    """
    Loads HotpotQA in distractor setting.
    split: "train" (90k) or "validation" (7.4k)
    max_samples: subset size for quick testing
    """
    print(f"Loading HotpotQA ({split})...")
    dataset = load_dataset("hotpot_qa", "distractor", split=split)

    if max_samples:
        dataset = dataset.select(range(max_samples))

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
