"""Combine the red and white UCI Wine Quality CSV files.

Importing this module does not read files or create output. Call
combine_wine_datasets() from code or run this file as a script.
"""

from __future__ import annotations

import csv
import math
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


REQUIRED_COLUMNS = {
    "fixed_acidity",
    "volatile_acidity",
    "citric_acid",
    "residual_sugar",
    "chlorides",
    "free_sulfur_dioxide",
    "total_sulfur_dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
    "quality",
}


def normalize_header(header: str) -> str:
    """Normalize a UCI CSV header while retaining its original meaning."""
    return header.strip().replace(" ", "_")


def read_csv_semicolon(filepath: str | Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read a semicolon-delimited UCI source file and normalize its headers."""
    source = Path(filepath)
    if not source.is_file():
        raise FileNotFoundError(f"Dataset file not found: {source}")

    with source.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, delimiter=";")
        raw_headers = reader.fieldnames
        if not raw_headers:
            raise ValueError(f"Dataset has no header row: {source}")

        headers = [normalize_header(header) for header in raw_headers]
        if any(not header for header in headers):
            raise ValueError(f"Dataset contains an empty column name: {source}")
        if len(headers) != len(set(headers)):
            raise ValueError(f"Dataset contains duplicate normalized headers: {source}")

        rows: list[dict[str, str]] = []
        for line_number, raw_row in enumerate(reader, start=2):
            if None in raw_row:
                raise ValueError(f"Unexpected extra column in {source} at line {line_number}")
            if any(value is None for value in raw_row.values()):
                raise ValueError(f"Missing column value in {source} at line {line_number}")
            row = {
                normalize_header(key): value.strip()
                for key, value in raw_row.items()
                if key is not None and value is not None
            }
            if any(value == "" for value in row.values()):
                raise ValueError(f"Empty value in {source} at line {line_number}")
            rows.append(row)

    if not rows:
        raise ValueError(f"Dataset contains no data rows: {source}")

    return headers, rows


def combine_wine_datasets(
    red_path: str | Path,
    white_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Validate, combine, and atomically write the two wine datasets."""
    red_headers, red_rows = read_csv_semicolon(red_path)
    white_headers, white_rows = read_csv_semicolon(white_path)

    if red_headers != white_headers:
        raise ValueError("Red and white wine datasets have different column headers.")
    if not REQUIRED_COLUMNS.issubset(red_headers):
        missing = sorted(REQUIRED_COLUMNS.difference(red_headers))
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")

    all_rows = red_rows + white_rows
    quality_counts: Counter[int] = Counter()
    normalized_rows: list[dict[str, str]] = []

    for index, row in enumerate(all_rows, start=2):
        normalized_row: dict[str, str] = {}
        try:
            raw_quality = float(row["quality"])
            if not math.isfinite(raw_quality) or not raw_quality.is_integer():
                raise ValueError("quality must be a finite integer")
            quality = int(raw_quality)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid quality label at combined row {index}.") from error

        for column in red_headers:
            if column == "quality":
                normalized_row[column] = str(quality)
                continue
            try:
                numeric_value = float(row[column])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid numeric value for {column!r} at combined row {index}."
                ) from error
            if not math.isfinite(numeric_value):
                raise ValueError(
                    f"Non-finite numeric value for {column!r} at combined row {index}."
                )
            # This matches the float-column representation in the committed CSV
            # (for example, 11 becomes 11.0 while quality remains an integer).
            normalized_row[column] = repr(numeric_value)

        normalized_rows.append(normalized_row)
        quality_counts[quality] += 1

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = temp_file.name
            writer = csv.DictWriter(temp_file, fieldnames=red_headers)
            writer.writeheader()
            writer.writerows(normalized_rows)

        os.replace(temp_path, destination)
        temp_path = None
    finally:
        if temp_path is not None:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass

    good = sum(count for label, count in quality_counts.items() if label >= 6)
    bad = sum(count for label, count in quality_counts.items() if label < 6)
    return {
        "rows": len(all_rows),
        "headers": red_headers,
        "quality_counts": dict(sorted(quality_counts.items())),
        "good": good,
        "bad": bad,
        "output_path": str(destination),
    }


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    raw_dir = base_dir / "data_raw"
    output_path = base_dir / "data" / "wine_quality.csv"
    result = combine_wine_datasets(
        raw_dir / "winequality-red.csv",
        raw_dir / "winequality-white.csv",
        output_path,
    )

    print(f"Berhasil! Total baris: {result['rows']}")
    print(f"Kolom: {result['headers']}")
    print("\nDistribusi quality:")
    for quality, count in result["quality_counts"].items():
        print(f"  {quality}: {count}")
    print("\nLabel binarisasi:")
    print(f"  good (>=6): {result['good']}")
    print(f"  bad  (<6) : {result['bad']}")
    print(f"\nDisimpan ke: {result['output_path']}")


if __name__ == "__main__":
    main()
