from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple


MergedRow = Tuple[float, Optional[float], Optional[float]]


def to_float(value: str, file_path: Path, row_number: int) -> float:
    """Convert a CSV cell to float with a clear error message."""
    try:
        return float(value.strip())
    except ValueError as exc:
        raise ValueError(
            f"Cannot convert value to float in {file_path} "
            f"at row {row_number}: {value!r}"
        ) from exc


def read_non_empty_rows(file_path: Path) -> List[Tuple[int, List[str]]]:
    """Read non-empty CSV rows and keep original row numbers."""
    rows: List[Tuple[int, List[str]]] = []

    with file_path.open("r", newline="") as file:
        reader = csv.reader(file)

        for row_number, row in enumerate(reader, start=1):
            cleaned_row = [cell.strip() for cell in row]

            if cleaned_row and any(cleaned_row):
                rows.append((row_number, cleaned_row))

    return rows


def read_material_csv(file_path: Path) -> List[MergedRow]:
    """
    Read one material CSV file and merge n and k data.

    Supported formats:

    1. Old two-block format:
        wl,n
        ...
        
        wl,k
        ...

    2. Already merged format:
        wl,n,k
        ...

    Returns:
        List of (wavelength, n, k), using the original wavelength order.
    """
    rows = read_non_empty_rows(file_path)

    if not rows:
        raise ValueError(f"Empty CSV file: {file_path}")

    _, first_row = rows[0]

    # Case 1: already merged format: wl,n,k
    if first_row == ["wl", "n", "k"]:
        merged_rows: List[MergedRow] = []

        for row_number, row in rows[1:]:
            if len(row) != 3:
                raise ValueError(
                    f"Invalid merged-format row in {file_path} "
                    f"at row {row_number}: {row}. Expected wl,n,k."
                )

            wavelength = to_float(row[0], file_path, row_number)
            n_value = to_float(row[1], file_path, row_number) if row[1] else None
            k_value = to_float(row[2], file_path, row_number) if row[2] else None

            merged_rows.append((wavelength, n_value, k_value))

        return merged_rows

    # Case 2: old two-block format: wl,n and wl,k
    n_data: Dict[float, float] = {}
    k_data: Dict[float, float] = {}
    wavelengths: List[float] = []
    seen_wavelengths = set()
    current_block: Optional[str] = None

    for row_number, row in rows:
        if row == ["wl", "n"]:
            current_block = "n"
            continue

        if row == ["wl", "k"]:
            current_block = "k"
            continue

        if len(row) != 2:
            raise ValueError(
                f"Invalid row in {file_path} at row {row_number}: {row}. "
                "Expected header wl,n / wl,k or two numeric columns."
            )

        if current_block not in {"n", "k"}:
            raise ValueError(
                f"Data row appears before wl,n or wl,k header in {file_path} "
                f"at row {row_number}: {row}"
            )

        wavelength = to_float(row[0], file_path, row_number)
        value = to_float(row[1], file_path, row_number)

        if wavelength not in seen_wavelengths:
            wavelengths.append(wavelength)
            seen_wavelengths.add(wavelength)

        if current_block == "n":
            n_data[wavelength] = value
        else:
            k_data[wavelength] = value

    if not n_data and not k_data:
        raise ValueError(f"No n or k data found in {file_path}")

    merged_rows = [
        (
            wavelength,
            n_data.get(wavelength),
            k_data.get(wavelength),
        )
        for wavelength in wavelengths
    ]

    return merged_rows


def write_merged_csv(output_path: Path, rows: List[MergedRow]) -> None:
    """Write merged material data in wl,n,k format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["wl", "n", "k"])

        for wavelength, n_value, k_value in rows:
            writer.writerow(
                [
                    wavelength,
                    "" if n_value is None else n_value,
                    "" if k_value is None else k_value,
                ]
            )


def merge_all_materials() -> None:
    """
    Merge all CSV files under jaxlayerlumos/materials and overwrite them in
    wl,n,k format.
    """
    repo_root = Path(__file__).resolve().parents[1]

    source_dir = repo_root / "jaxlayerlumos" / "materials"

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    csv_files = sorted(source_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {source_dir}")

    print(f"Source directory: {source_dir}")
    print("Output mode: overwrite source CSV files")
    print(f"Number of CSV files: {len(csv_files)}")
    print("-" * 100)

    success_count = 0
    errors = []

    for source_path in csv_files:
        output_path = source_path

        try:
            merged_rows = read_material_csv(source_path)

            write_merged_csv(output_path, merged_rows)

            success_count += 1
            print(
                f"[OK] {source_path.name:<40} "
                f"rows={len(merged_rows):>5} "
                f"-> {output_path.relative_to(repo_root)}"
            )

        except Exception as exc:
            errors.append((source_path, exc))
            print(f"[ERROR] {source_path.name:<40} {exc}")

    print("-" * 100)
    print(f"Successfully merged: {success_count}/{len(csv_files)}")

    if errors:
        print(f"Errors: {len(errors)}")
        for file_path, exc in errors:
            print(f"  - {file_path.name}: {exc}")
        raise SystemExit(1)

    print(f"All merged files have overwritten source files in: {source_dir}")


if __name__ == "__main__":
    merge_all_materials()

