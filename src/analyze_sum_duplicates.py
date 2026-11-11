"""Inspect exact SUM text duplicates without changing or saving any data.

Run from the project root: python src/analyze_sum_duplicates.py
Equality is case-sensitive and includes all whitespace; no normalization is used.
"""

import csv
from collections import defaultdict
from itertools import combinations
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed" / "sum"
SPLITS = ("train", "validation", "test")


def load_occurrences():
    """Index every original CSV occurrence by its unchanged text value."""
    occurrences = defaultdict(list)
    within = {}
    for split in SPLITS:
        local = defaultdict(list)
        # Read-only access. csv preserves quoted commas and embedded newlines.
        with (DATA_DIR / f"{split}.csv").open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not {"pair_id", "text", "label"} <= set(reader.fieldnames or []):
                raise ValueError(f"{split}.csv is missing required columns.")
            for row in reader:
                if row["label"] not in {"ANSWERABLE", "UNANSWERABLE"}:
                    raise ValueError(f"Unexpected label in {split}: {row['label']!r}")
                record = {
                    "split": split,
                    "pair_id": int(row["pair_id"]),
                    "label": row["label"],
                }
                occurrences[row["text"]].append(record)
                local[row["text"]].append(record)
        within[split] = local
    return occurrences, within


def has_conflict(records):
    """A conflict means the exact same text occurs under both class labels."""
    return len({record["label"] for record in records}) > 1


def main():
    occurrences, within = load_occurrences()
    # A text counts once globally even if repeated in all three splits or rows.
    cross = {
        text: records for text, records in occurrences.items()
        if len({record["split"] for record in records}) > 1
    }
    conflicts = {text for text, records in cross.items() if has_conflict(records)}
    different_pairs = {
        text for text, records in cross.items()
        if len({record["pair_id"] for record in records}) > 1
    }

    print("SUM exact duplicate-text analysis (read-only)")
    print("Counts refer to distinct text values, not occurrence pairs.")
    print(f"Total unique cross-split duplicate texts: {len(cross):,}")
    for left, right in combinations(SPLITS, 2):
        shared = set(within[left]) & set(within[right])
        print(f"{left} <-> {right}: {len(shared):,}")
    print(f"Cross-split duplicate texts with the same label everywhere: {len(cross) - len(conflicts):,}")
    print(f"Cross-split duplicate texts with conflicting labels: {len(conflicts):,}")
    print(f"Cross-split duplicate texts involving different pair_ids: {len(different_pairs):,}")

    print("\nWithin-split exact duplicates:")
    for split, texts in within.items():
        repeated = {text: rows for text, rows in texts.items() if len(rows) > 1}
        # Distinguish unique repeated strings, all their rows, and extra copies.
        rows = sum(len(records) for records in repeated.values())
        conflicting = sum(has_conflict(records) for records in repeated.values())
        print(f"{split}: {len(repeated):,} unique duplicate texts; "
              f"{rows:,} rows involved; {rows - len(repeated):,} extra copies; "
              f"{conflicting:,} texts with conflicting labels")

    # Deterministic illustrative sample: alternate conflicting and consistent
    # cases when available, then fill remaining slots. This is not a random or
    # statistically representative sample. Display ALL occurrences per text.
    groups = [iter(sorted(conflicts)), iter(sorted(set(cross) - conflicts))]
    selected = []
    while len(selected) < min(10, len(cross)):
        for group in groups:
            text = next(group, None)
            if text is not None:
                selected.append(text)
            if len(selected) == min(10, len(cross)):
                break

    print(f"\n{len(selected)} illustrative cross-split duplicate examples:")
    print("Text is shown with repr(), exposing whitespace without normalizing it.")
    for index, text in enumerate(selected, 1):
        records = cross[text]
        print(f"\nExample {index}: label conflict={has_conflict(records)}; "
              f"different pair_ids={text in different_pairs}")
        print(f"text: {text!r}")
        for record in sorted(records, key=lambda r: (SPLITS.index(r["split"]), r["pair_id"], r["label"])):
            print(f"  split={record['split']} | pair_id={record['pair_id']} | label={record['label']}")

    print("\nIdentical text under different pair_ids can cross a pair-level split.")
    print("Text equality alone does not establish why the publisher repeated it.")
    print("No dataset files were changed or written.")


if __name__ == "__main__":
    main()
