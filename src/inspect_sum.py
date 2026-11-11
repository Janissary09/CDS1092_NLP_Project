"""Load and inspect the official Synthetic Unanswerable Math (SUM) dataset.

Run from the project root: python src/inspect_sum.py
Source: https://huggingface.co/datasets/lime-nlp/Synthetic_Unanswerable_Math

Hugging Face may download and cache the source data when this script runs.
The script does not transform the data or save a separate dataset copy.
"""

from pprint import pprint


DATASET_ID = "lime-nlp/Synthetic_Unanswerable_Math"
# File-to-split mapping from the publisher's README configuration.
SOURCE_FILES = {
    "train": "synthetic_unanswerable_math.parquet",
    "test": "test.parquet",
}


def main():
    # Import here so a missing dependency produces an understandable message.
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "The Hugging Face 'datasets' library is required. "
            "Run this script in an environment where it is installed."
        ) from exc

    # Load ONE file per call using the generic Parquet loader. Loading the
    # repository configuration as a whole tries to share a schema across files,
    # but the publisher's train and test files have different columns.
    # These labels refer to published sources; no new splits are created.
    for split_name, filename in SOURCE_FILES.items():
        source_url = (
            f"https://huggingface.co/datasets/{DATASET_ID}/resolve/main/{filename}"
        )
        print(f"\n{'=' * 72}\nPublisher split: {split_name}")
        print(f"Source file: {filename}")
        print(f"URL: {source_url}", flush=True)
        split = load_dataset(
            "parquet",
            data_files={split_name: source_url},
            split=split_name,
        )
        print(f"Number of rows: {split.num_rows}")
        print("Column names:", split.column_names)
        print("Features / schema:")
        pprint(split.features)

        # Inspect the first two rows without changing their order or contents.
        # min() also handles a split that contains fewer than two rows.
        for row_index in range(min(2, split.num_rows)):
            row = split[row_index]
            print(f"\nExample {row_index + 1} (row index {row_index}):")

            # Show every field with its exact original name and value. In
            # particular, do not infer what 'prompt' or 'ground_truth' means.
            for field, value in row.items():
                print(f"\n{field}:")
                pprint(value, sort_dicts=False)


# Importing this file does not load the dataset; direct execution does.
if __name__ == "__main__":
    main()
