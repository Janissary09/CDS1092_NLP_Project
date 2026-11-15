"""Prepare the unchanged official UMWP external evaluation benchmark.

Run from the project root: python src/08_prepare_umwp.py
UMWP must never influence training, tuning, checkpoint selection or early stopping.
Models are trained and selected using SUM only. No UMWP split is created.
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data/raw/umwp/StandardDataset.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "data/processed/umwp/test.csv"
REQUIRED_COLUMNS = {"id", "question", "answerable", "category", "source"}
OUTPUT_COLUMNS = ["id", "text", "label", "category", "source"]
LABEL_MAPPING = {True: "ANSWERABLE", False: "UNANSWERABLE"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def question_statistics(frame, text_column, label_column):
    """Count exact texts and conflicts without normalizing or removing rows."""
    counts = frame[text_column].value_counts()
    duplicates = counts[counts > 1]
    duplicate_rows = frame.loc[frame[text_column].isin(duplicates.index)]
    label_counts = duplicate_rows.groupby(text_column)[label_column].nunique(dropna=False)
    return {
        "unique_question_texts": len(counts),
        "duplicated_question_texts": len(duplicates),
        "conflicting_duplicated_question_texts": int((label_counts > 1).sum()),
    }


def validate_text(text):
    require(text.notna().all(), "Missing question/text values detected.")
    require(text.map(lambda value: isinstance(value, str)).all(),
            "Every question/text value must be a string.")
    require(text.map(lambda value: bool(value.strip())).all(),
            "Empty or whitespace-only question/text values detected.")


def main():
    # Object dtype preserves category codes and nulls without numeric coercion.
    raw = pd.read_json(INPUT_PATH, lines=True, encoding="utf-8-sig",
                       dtype={"id": object, "question": object, "answerable": object,
                              "category": object, "source": object})
    require(REQUIRED_COLUMNS.issubset(raw.columns),
            f"Missing required columns: {sorted(REQUIRED_COLUMNS - set(raw.columns))}")
    require(len(raw) == 5200, "Expected exactly 5,200 official raw rows.")
    require(raw["id"].notna().all() and raw["id"].is_unique,
            "Raw IDs must be unique and non-null.")
    validate_text(raw["question"])
    require(raw["answerable"].map(lambda value: isinstance(value, bool)).all(),
            "answerable must contain only boolean True/False values.")

    raw_statistics = question_statistics(raw, "question", "answerable")
    print("Unique question texts:", raw_statistics["unique_question_texts"])
    print(f"WARNING: {raw_statistics['duplicated_question_texts']} exact duplicated "
          "question texts detected; all rows retained.")
    print(f"WARNING: {raw_statistics['conflicting_duplicated_question_texts']} duplicated "
          "question texts have conflicting answerability labels; labels unchanged.")

    # Select metadata and text directly, preserving values, indices and row order.
    # Neither answer nor relevant_ids is used in the processed dataset.
    processed = raw[["id", "question", "category", "source"]].rename(
        columns={"question": "text"}).copy()
    processed["label"] = raw["answerable"].map(LABEL_MAPPING)
    processed = processed[OUTPUT_COLUMNS]

    require(len(processed) == 5200, "Expected exactly 5,200 processed rows.")
    require(processed["id"].notna().all() and processed["id"].is_unique,
            "Processed IDs must remain unique and non-null.")
    validate_text(processed["text"])
    require(processed["label"].isin(LABEL_MAPPING.values()).all(), "Invalid processed labels.")
    label_counts = processed["label"].value_counts()
    require(label_counts.get("ANSWERABLE", 0) == 2600
            and label_counts.get("UNANSWERABLE", 0) == 2600,
            "Expected exactly 2,600 examples per class.")
    processed_statistics = question_statistics(processed, "text", "label")
    require(processed_statistics["unique_question_texts"] == 5190,
            "Expected exactly 5,190 unique question texts.")
    require(processed_statistics == raw_statistics, "Duplicate/conflict counts changed.")
    for original, output in [("id", "id"), ("question", "text"),
                             ("category", "category"), ("source", "source")]:
        require(raw[original].equals(processed[output]), f"Original {original} values/order changed.")
    require(processed["label"].equals(raw["answerable"].map(LABEL_MAPPING)),
            "Answerability labels changed.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print("\nValidation summary")
    print("Raw rows:", len(raw))
    print("Processed rows:", len(processed))
    print("ANSWERABLE count:", label_counts["ANSWERABLE"])
    print("UNANSWERABLE count:", label_counts["UNANSWERABLE"])
    for name, count in processed_statistics.items():
        print(f"{name.replace('_', ' ').capitalize()}: {count}")
    print("Sources (counts):")
    print(processed["source"].value_counts(dropna=False).to_string())
    print("Category distribution (original codes; null retained):")
    print(processed["category"].value_counts(dropna=False).to_string())
    print("Output path:", OUTPUT_PATH)
    print("\nUMWP preparation completed successfully.")
    print("Official 5,200-instance benchmark preserved unchanged.")
    print("Dataset ready for external evaluation.")


if __name__ == "__main__":
    main()
