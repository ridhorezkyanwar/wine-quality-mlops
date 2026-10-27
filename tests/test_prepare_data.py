from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_data import combine_wine_datasets  # noqa: E402


HEADERS = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
    "quality",
]


def write_source(path: Path, headers: list[str], rows: list[list[str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(headers)
        writer.writerows(rows)


def row(quality: str) -> list[str]:
    return ["7.4", "0.7", "0.0", "1.9", "0.076", "11", "34", "0.9978", "3.51", "0.56", "9.4", quality]


def test_combines_red_and_white_rows_and_reports_binary_labels(tmp_path: Path) -> None:
    red = tmp_path / "red.csv"
    white = tmp_path / "white.csv"
    output = tmp_path / "nested" / "wine_quality.csv"
    write_source(red, HEADERS, [row("5"), row("6")])
    write_source(white, HEADERS, [row("7")])

    result = combine_wine_datasets(red, white, output)

    assert result["rows"] == 3
    assert result["good"] == 2
    assert result["bad"] == 1
    assert result["quality_counts"] == {5: 1, 6: 1, 7: 1}
    with output.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == [header.replace(" ", "_") for header in HEADERS]
        assert [item["quality"] for item in reader] == ["5", "6", "7"]


def test_does_not_overwrite_output_when_headers_do_not_match(tmp_path: Path) -> None:
    red = tmp_path / "red.csv"
    white = tmp_path / "white.csv"
    output = tmp_path / "existing.csv"
    write_source(red, HEADERS, [row("5")])
    mismatched_headers = HEADERS[:-1] + ["rating"]
    write_source(white, mismatched_headers, [row("7")])
    output.write_text("keep existing output\n", encoding="utf-8")

    with pytest.raises(ValueError, match="different column headers"):
        combine_wine_datasets(red, white, output)

    assert output.read_text(encoding="utf-8") == "keep existing output\n"


def test_rejects_empty_dataset_without_overwriting_output(tmp_path: Path) -> None:
    red = tmp_path / "red.csv"
    white = tmp_path / "white.csv"
    output = tmp_path / "existing.csv"
    red.write_text(";".join(HEADERS) + "\n", encoding="utf-8")
    write_source(white, HEADERS, [row("7")])
    output.write_text("safe prior output\n", encoding="utf-8")

    with pytest.raises(ValueError, match="no data rows"):
        combine_wine_datasets(red, white, output)

    assert output.read_text(encoding="utf-8") == "safe prior output\n"


def test_rejects_missing_quality_column(tmp_path: Path) -> None:
    red = tmp_path / "red.csv"
    white = tmp_path / "white.csv"
    headers = HEADERS[:-1]
    values = row("5")[:-1]
    write_source(red, headers, [values])
    write_source(white, headers, [values])

    with pytest.raises(ValueError, match="missing required columns"):
        combine_wine_datasets(red, white, tmp_path / "out.csv")
