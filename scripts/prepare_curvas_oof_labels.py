"""Validate extracted CURVAS cases and create pancreas majority-vote masks.

This script deliberately does not create an nnU-Net training dataset: the
patient-level train/calibration/test split must be frozen first.
"""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np


EXPECTED_COUNTS = {"training_set": 20, "validation_set": 5, "testing_set": 65}
EXPECTED_FILES = {"image.nii.gz", *(f"annotation_{i}.nii.gz" for i in range(1, 4))}
LABEL_VALUES = {0, 1, 2, 3}  # CURVAS: background, pancreas, kidney, liver
# Source header anomaly confirmed by index-space overlap with the other raters.
# The derived majority mask remains provisional until manual image-overlay review.
INDEX_GRID_HEADER_EXCEPTION = ("testing_set", "UKCHLL007", 2)


def collect_cases(extracted_root: Path) -> list[tuple[str, Path]]:
    cases: list[tuple[str, Path]] = []
    seen_ids: set[str] = set()
    for source_split, expected_count in EXPECTED_COUNTS.items():
        split_dir = extracted_root / source_split
        if not split_dir.is_dir():
            raise ValueError(f"Missing extracted directory: {split_dir}")
        split_cases = sorted(path for path in split_dir.iterdir() if path.is_dir())
        if len(split_cases) != expected_count:
            raise ValueError(
                f"{source_split}: expected {expected_count} cases, got {len(split_cases)}"
            )
        for case_dir in split_cases:
            actual_files = {path.name for path in case_dir.iterdir() if path.is_file()}
            if actual_files != EXPECTED_FILES:
                raise ValueError(
                    f"{case_dir}: expected {sorted(EXPECTED_FILES)}, got {sorted(actual_files)}"
                )
            if case_dir.name in seen_ids:
                raise ValueError(f"Repeated patient ID across archives: {case_dir.name}")
            seen_ids.add(case_dir.name)
            cases.append((source_split, case_dir))
    return cases


def same_grid(reference: nib.spatialimages.SpatialImage, other: nib.spatialimages.SpatialImage) -> bool:
    return (
        reference.shape == other.shape
        and np.all(np.isfinite(other.affine))
        and np.allclose(reference.affine, other.affine, rtol=1e-5, atol=1e-3)
    )


def raw_value_counts(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Count stored integer codes without sorting hundreds of millions of voxels."""
    if array.dtype == np.dtype("int16") and array.dtype.isnative:
        bins = np.bincount(array.view(np.uint16).ravel(), minlength=65536)
        occupied = np.flatnonzero(bins)
        return occupied.astype(np.uint16).view(np.int16), bins[occupied]
    return np.unique(array, return_counts=True)


def save_mask_atomic(mask: np.ndarray, reference: nib.Nifti1Image, destination: Path) -> None:
    header = reference.header.copy()
    header.set_data_dtype(np.uint8)
    header.set_slope_inter(1, 0)
    output = nib.Nifti1Image(mask, reference.affine, header=header)
    for name in ("qform", "sform"):
        matrix, code = getattr(reference, f"get_{name}")(coded=True)
        if matrix is not None and code:
            getattr(output, f"set_{name}")(matrix, int(code))

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=".nii.gz", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        nib.save(output, str(temporary))
        written = nib.load(str(temporary))
        if not same_grid(reference, written) or written.get_data_dtype() != np.uint8:
            raise ValueError(f"Written mask geometry or dtype changed: {destination}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def process_case(source_split: str, case_dir: Path, output_dir: Path) -> dict[str, object]:
    case_id = case_dir.name
    image = nib.load(str(case_dir / "image.nii.gz"))
    if len(image.shape) != 3 or not np.all(np.isfinite(image.affine)):
        raise ValueError(f"{case_id}: CT must have a finite 3D geometry")

    votes = np.zeros(image.shape, dtype=np.uint8)
    expert_voxels: list[int] = []
    header_exception_used = False
    for index in range(1, 4):
        label_path = case_dir / f"annotation_{index}.nii.gz"
        label = nib.load(str(label_path))
        grid_matches = same_grid(image, label)
        header_exception = (source_split, case_id, index) == INDEX_GRID_HEADER_EXCEPTION
        if not grid_matches and not header_exception:
            raise ValueError(f"{case_id}: annotation_{index} does not match the CT grid")
        if not grid_matches and label.shape != image.shape:
            raise ValueError(f"{case_id}: header exception cannot fix a shape mismatch")
        proxy = label.dataobj
        array = np.asanyarray(proxy.get_unscaled())
        raw_values, counts = raw_value_counts(array)
        unique_values = raw_values.astype(np.float64) * proxy.slope + proxy.inter
        nearest_labels = np.rint(unique_values)
        if not np.all(np.isfinite(unique_values)) or np.any(
            np.abs(unique_values - nearest_labels) > 1e-3
        ):
            raise ValueError(f"{case_id}: annotation_{index} has non-integer labels")
        observed = set(nearest_labels.astype(np.int16).tolist())
        if not observed.issubset(LABEL_VALUES):
            raise ValueError(f"{case_id}: annotation_{index} has invalid labels {observed}")
        pancreas_voxels = int(counts[nearest_labels == 1].sum())
        expert_voxels.append(pancreas_voxels)
        pancreas_codes = raw_values[nearest_labels == 1]
        if len(pancreas_codes) == 1:
            pancreas_mask = array == pancreas_codes[0]
        elif len(pancreas_codes) > 1:
            pancreas_mask = np.isin(array, pancreas_codes)
        else:
            pancreas_mask = np.zeros(image.shape, dtype=bool)
        if not grid_matches:
            first_expert_voxels = expert_voxels[0]
            overlap = int(np.count_nonzero((votes > 0) & pancreas_mask))
            index_dice = 2 * overlap / (first_expert_voxels + pancreas_voxels)
            if index_dice < 0.8:
                raise ValueError(f"{case_id}: header exception failed index Dice check")
            header_exception_used = True
        votes += pancreas_mask
        del array, label, pancreas_mask

    majority = (votes >= 2).astype(np.uint8)
    majority_voxels = int(np.count_nonzero(majority))
    disagreement_voxels = int(np.count_nonzero((votes == 1) | (votes == 2)))
    if majority_voxels == 0:
        raise ValueError(f"{case_id}: empty majority-vote pancreas mask")
    destination = output_dir / f"{case_id}.nii.gz"
    save_mask_atomic(majority, image, destination)
    return {
        "source_split": source_split,
        "case_id": case_id,
        "image": str(case_dir / "image.nii.gz"),
        "majority_mask": str(destination),
        "shape": "x".join(map(str, image.shape)),
        "spacing_mm": "x".join(f"{value:.6g}" for value in image.header.get_zooms()[:3]),
        "expert_1_voxels": expert_voxels[0],
        "expert_2_voxels": expert_voxels[1],
        "expert_3_voxels": expert_voxels[2],
        "majority_voxels": majority_voxels,
        "disagreement_voxels": disagreement_voxels,
        "qc_flag": (
            "REVIEW_HEADER_MISMATCH" if header_exception_used else
            "REVIEW_EMPTY_EXPERT" if 0 in expert_voxels else
            "PASS_AUTOMATED"
        ),
    }


def write_manifest(rows: list[dict[str, object]], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=".csv", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extracted-root", type=Path, default=Path("data/extracted/curvas"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/derived/curvas/pancreas_majority_vote")
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/manifests/curvas_oof_label_manifest.csv")
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    cases = collect_cases(args.extracted_root)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_case, source_split, case_dir, args.output_dir): case_dir.name
            for source_split, case_dir in cases
        }
        for number, future in enumerate(as_completed(futures), 1):
            row = future.result()
            rows.append(row)
            print(f"[{number:02d}/{len(cases)}] {futures[future]}: {row['qc_flag']}", flush=True)
    split_order = {name: index for index, name in enumerate(EXPECTED_COUNTS)}
    rows.sort(key=lambda row: (split_order[row["source_split"]], row["case_id"]))
    write_manifest(rows, args.manifest)
    print(f"Created {len(rows)} majority-vote masks and {args.manifest}")


if __name__ == "__main__":
    main()
