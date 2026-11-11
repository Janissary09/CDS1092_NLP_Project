"""Prepare the final SUM split with pair and exact-text isolation.

Run: python src/prepare_sum.py
Original row indices are pair IDs. No text normalization or deletion is used.
"""

import csv
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
import random


SOURCE_URL = (
    "https://huggingface.co/datasets/lime-nlp/Synthetic_Unanswerable_Math/"
    "resolve/main/synthetic_unanswerable_math.parquet"
)
SOURCE_PAIRS = 36_480
SEED = 42
TARGETS = {"train": 25_536, "validation": 5_472, "test": 5_472}
FIELDS = {"ANSWERABLE": "answerable_question", "UNANSWERABLE": "unanswerable_question"}
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data/processed/sum"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def build_groups(source):
    """Union pairs sharing either question, including transitive connections."""
    parent = list(range(len(source)))
    sizes = [1] * len(source)

    def find(pair_id):
        # Path compression keeps repeated root lookups efficient.
        while parent[pair_id] != pair_id:
            parent[pair_id] = parent[parent[pair_id]]
            pair_id = parent[pair_id]
        return pair_id

    def union(left, right):
        left, right = find(left), find(right)
        if left == right:
            return
        if sizes[left] < sizes[right]:
            left, right = right, left
        parent[right] = left
        sizes[left] += sizes[right]

    first_pair = {}
    for pair_id, row in enumerate(source):
        for field in FIELDS.values():
            text = row[field]
            require(isinstance(text, str), f"Pair {pair_id}: {field} must be a string.")
            if text in first_pair:
                union(pair_id, first_pair[text])
            else:
                first_pair[text] = pair_id

    groups = defaultdict(list)
    for pair_id in range(len(source)):
        groups[find(pair_id)].append(pair_id)
    return list(groups.values())


def assign_groups(groups):
    """Place large groups first, filling target deficits without breaking groups.

    A seeded shuffle breaks equal-size ties reproducibly. Each group goes into
    the split with the largest remaining pair deficit. This greedy heuristic
    aims for the requested counts; group integrity always takes precedence.
    Small groups placed last allow fine adjustment near the targets.
    """
    ordered = list(groups)
    random.Random(SEED).shuffle(ordered)
    ordered.sort(key=len, reverse=True)  # Stable sort retains seeded tie order.
    assigned = {name: [] for name in TARGETS}
    for group in ordered:
        name = max(TARGETS, key=lambda name: TARGETS[name] - len(assigned[name]))
        assigned[name].extend(group)
    return assigned


def expand(source, assigned):
    """Expand only after assigning every complete connected group."""
    splits = {}
    for name, pair_ids in assigned.items():
        rows = []
        for pair_id in pair_ids:
            for label, field in FIELDS.items():
                rows.append({"pair_id": pair_id, "text": source[pair_id][field], "label": label})
        random.Random(SEED).shuffle(rows)
        splits[name] = rows
    return splits


def validate(source, groups, assigned, splits):
    """Stop on any loss, corruption, group break, imbalance, or leakage."""
    require(len(source) == SOURCE_PAIRS, "Incorrect source pair count.")
    pair_sets = {}
    text_sets = {}
    for name, rows in splits.items():
        pairs = Counter(row["pair_id"] for row in rows)
        labels = Counter(row["label"] for row in rows)
        require(bool(pairs), f"{name}: empty split.")
        require(set(pairs) == set(assigned[name]), f"{name}: assigned IDs do not match examples.")
        require(len(assigned[name]) == len(pairs), f"{name}: repeated assigned pair ID.")
        require(all(count == 2 for count in pairs.values()), f"{name}: pair must appear twice.")
        require(labels == Counter({label: len(pairs) for label in FIELDS}), f"{name}: class imbalance.")
        seen = set()
        for row in rows:
            pair_id, label = row["pair_id"], row["label"]
            require((pair_id, label) not in seen, f"{name}: repeated pair/label.")
            seen.add((pair_id, label))
            require(row["text"] == source[pair_id][FIELDS[label]], f"{name}: text changed.")
        pair_sets[name] = set(pairs)
        text_sets[name] = {row["text"] for row in rows}

    for left, right in combinations(TARGETS, 2):
        require(pair_sets[left].isdisjoint(pair_sets[right]), f"Pair leakage: {left}/{right}.")
        require(text_sets[left].isdisjoint(text_sets[right]), f"Text leakage: {left}/{right}.")
    require(set().union(*pair_sets.values()) == set(range(SOURCE_PAIRS)), "Source pairs lost or added.")
    require(sum(map(len, splits.values())) == 72_960, "Incorrect total example count.")
    owner = {pair_id: name for name, ids in pair_sets.items() for pair_id in ids}
    require(all(len({owner[pair_id] for pair_id in group}) == 1 for group in groups),
            "A connected group was broken.")


def main():
    from datasets import load_dataset

    print(f"Loading only: {SOURCE_URL}", flush=True)
    source = load_dataset("parquet", data_files={"train": SOURCE_URL}, split="train")
    require(len(source) == SOURCE_PAIRS, f"Expected {SOURCE_PAIRS} source pairs; got {len(source)}.")
    require(set(FIELDS.values()) | {"ground_truth"} <= set(source.column_names),
            "Missing required source columns.")
    groups = build_groups(source)
    assigned = assign_groups(groups)
    splits = expand(source, assigned)
    validate(source, groups, assigned, splits)

    # Write the final dataset directory only after all checks pass.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        with (OUTPUT_DIR / f"{name}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["pair_id", "text", "label"])
            writer.writeheader()
            writer.writerows(rows)

    distribution = Counter(map(len, groups))
    print(f"\nSource pairs: {len(source):,}; seed: {SEED}")
    print(f"Connected groups: {len(groups):,}; largest: {max(distribution):,} pairs")
    print(f"Groups with more than one pair: {sum(count for size, count in distribution.items() if size > 1):,}")
    print("Group size distribution (pairs per group -> number of groups):")
    for size, count in sorted(distribution.items()):
        print(f"  {size} -> {count:,}")
    print("\nSplit | target % | actual % | pairs | examples | ANSWERABLE | UNANSWERABLE")
    for name, rows in splits.items():
        pairs = len(assigned[name])
        labels = Counter(row["label"] for row in rows)
        print(f"{name} | {100 * TARGETS[name] / SOURCE_PAIRS:.2f}% | "
              f"{100 * pairs / SOURCE_PAIRS:.4f}% | {pairs:,} | {len(rows):,} | "
              f"{labels['ANSWERABLE']:,} | {labels['UNANSWERABLE']:,}")
    print("Pair leakage: 0 (PASS); cross-split exact-text leakage: 0 (PASS)")
    print("Complete groups, unchanged texts, two examples per pair, 50/50 classes: PASS")
    print("All 36,480 pairs and 72,960 examples retained: PASS")
    for name in TARGETS:
        path = OUTPUT_DIR / f"{name}.csv"
        size = path.stat().st_size
        print(f"{path}: {size:,} bytes ({size / 1024 ** 2:.2f} MiB)")


if __name__ == "__main__":
    main()
