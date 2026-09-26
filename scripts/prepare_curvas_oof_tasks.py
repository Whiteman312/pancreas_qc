"""Build five strictly separated 48-patient nnU-Net raw tasks from frozen folds.

Run from the project root. No fingerprinting, planning, preprocessing or training
is performed. Existing mismatched links/metadata are never overwritten.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
from collections import Counter
from pathlib import Path

import yaml

from export_curvas_nnunet import (
    assert_same_geometry,
    ensure_text,
    expected_export,
    read_csv,
    sha256,
)
from freeze_curvas_split import build_split


def dataset_metadata(count: int) -> dict:
    return {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "pancreas": 1},
        "numTraining": count,
        "file_ending": ".nii.gz",
    }


def validate_fold_protocol(split: dict) -> None:
    if split["counts"] != {"train": 60, "calibration": 10, "test": 20}:
        raise ValueError("Expected frozen 60/10/20 protocol")
    partitions = split["splits"]
    if set(partitions) != {"train", "calibration", "test"}:
        raise ValueError("Unexpected patient partitions")
    all_ids = [case for cases in partitions.values() for case in cases]
    if len(all_ids) != 90 or len(set(all_ids)) != 90:
        raise ValueError("Patient partitions overlap or contain duplicate/missing IDs")
    for name, expected in split["counts"].items():
        if len(partitions[name]) != expected:
            raise ValueError(f"Incorrect partition size: {name}")
    folds = split["folds"]
    if len(folds) != 5 or {fold["fold"] for fold in folds} != set(range(5)):
        raise ValueError("Expected exactly five distinct folds numbered 0..4")
    train_ids = set(partitions["train"])
    train_occurrences, held_out_occurrences = Counter(), Counter()
    for fold in folds:
        train, held_out = fold["train"], fold["held_out"]
        if len(train) != 48 or len(set(train)) != 48:
            raise ValueError(f"Fold {fold['fold']} must contain 48 unique training patients")
        if len(held_out) != 12 or len(set(held_out)) != 12:
            raise ValueError(f"Fold {fold['fold']} must contain 12 unique held-out patients")
        if set(train) & set(held_out) or set(train) | set(held_out) != train_ids:
            raise ValueError(f"Fold {fold['fold']} has overlap or foreign patients")
        train_occurrences.update(train)
        held_out_occurrences.update(held_out)
    if train_occurrences != Counter({case: 4 for case in train_ids}):
        raise ValueError("Every training patient must occur in exactly four training tasks")
    if held_out_occurrences != Counter({case: 1 for case in train_ids}):
        raise ValueError("Every training patient must be held out exactly once")


def check_directory_names(directory: Path, expected: set[str]) -> None:
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError(f"Expected a real dataset directory: {directory}")
    actual = {entry.name for entry in directory.iterdir()}
    if actual != expected:
        raise ValueError(
            f"Unexpected dataset entries in {directory}: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def check_link(target: Path, source: Path) -> None:
    if not target.is_symlink():
        raise ValueError(f"Expected relative symbolic link: {target}")
    expected = os.path.relpath(source, start=target.parent)
    if os.readlink(target) != expected or target.resolve(strict=True) != source.resolve(strict=True):
        raise ValueError(f"Link points to the wrong source: {target}")
    if not target.is_file():
        raise ValueError(f"Linked CT/mask is not a file: {target}")


def ensure_link(target: Path, source: Path) -> None:
    if target.exists() or target.is_symlink():
        check_link(target, source)
        return
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(os.path.relpath(source, start=target.parent))
    check_link(target, source)


def preflight_task(dataset_dir: Path, fold: dict, base_rows: dict[str, dict[str, str]]) -> None:
    """Reject collisions before modifying an existing or interrupted export."""
    if not dataset_dir.exists() and not dataset_dir.is_symlink():
        return
    if dataset_dir.is_symlink() or not dataset_dir.is_dir():
        raise ValueError(f"Task root is not a real directory: {dataset_dir}")
    if {entry.name for entry in dataset_dir.iterdir()} - {"imagesTr", "labelsTr", "dataset.json"}:
        raise ValueError(f"Unexpected existing entries in task: {dataset_dir}")
    metadata = dataset_dir / "dataset.json"
    if metadata.is_symlink():
        raise ValueError(f"Task metadata must not be a symbolic link: {metadata}")
    if metadata.exists() and metadata.read_text(encoding="utf-8") != json.dumps(dataset_metadata(48), indent=2) + "\n":
        raise ValueError(f"Existing task metadata differs: {metadata}")
    for directory_name, suffix, source_field in (
        ("imagesTr", "_0000.nii.gz", "export_image"),
        ("labelsTr", ".nii.gz", "export_mask"),
    ):
        directory = dataset_dir / directory_name
        if not directory.exists() and not directory.is_symlink():
            continue
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError(f"Task input folder is not a real directory: {directory}")
        expected = {case + suffix: Path(base_rows[case][source_field]) for case in fold["train"]}
        for entry in directory.iterdir():
            if entry.name not in expected:
                raise ValueError(f"Foreign patient/file in existing task: {entry}")
            check_link(entry, expected[entry.name])


def load_inputs(settings: dict) -> tuple[dict, dict, Path, dict[str, dict[str, str]]]:
    split, source_rows, base_dir, base_manifest = expected_export(settings)
    options = settings["nnunet"]["fold_tasks"]
    if sha256(Path(settings["split_manifest"])) != options["split_sha256"]:
        raise ValueError("Frozen patient split SHA-256 differs from the locked configuration")
    if build_split(settings, Path(settings["manifest"]), Path(settings["dispositions"])) != split:
        raise ValueError("Frozen split differs from its original configuration and source lists")
    validate_fold_protocol(split)
    if options["link_mode"] != "relative_symlink" or options["training_fold"] != "all":
        raise ValueError("Fold tasks require relative links and training_fold='all'")
    tasks = options["datasets"]
    if len(tasks) != 5 or {task["fold"] for task in tasks} != set(range(5)):
        raise ValueError("Configure exactly one task for each frozen fold")
    ids = [task["dataset_id"] for task in tasks]
    if len(set(ids)) != 5 or settings["nnunet"]["dataset_id"] in ids:
        raise ValueError("Dataset IDs must be distinct and must not reuse the staging ID")
    raw_root = Path(settings["nnunet"]["raw_root"])
    for task in tasks:
        if not isinstance(task["dataset_id"], int) or not 1 <= task["dataset_id"] <= 999:
            raise ValueError("Dataset IDs must be integers between 1 and 999")
        if task["dataset_name"] != f"CURVASPancreasOOF_Fold{task['fold']}":
            raise ValueError("Unexpected fold dataset name")
        prefix = f"Dataset{task['dataset_id']:03d}_"
        name = prefix + task["dataset_name"]
        collisions = [path for path in raw_root.glob(prefix + "*") if path.name != name]
        if collisions:
            raise ValueError(f"Dataset ID collision: {collisions}")

    train_ids = set(split["splits"]["train"])
    check_directory_names(base_dir / "imagesTr", {f"{case}_0000.nii.gz" for case in train_ids})
    check_directory_names(base_dir / "labelsTr", {f"{case}.nii.gz" for case in train_ids})
    if (base_dir / "imagesTs").exists():
        raise ValueError("Staging input must contain training patients only")
    if json.loads((base_dir / "dataset.json").read_text(encoding="utf-8")) != dataset_metadata(60):
        raise ValueError("Staging dataset.json differs from the verified single-channel CT task")
    base_rows = read_csv(base_manifest)
    by_id = {row["case_id"]: row for row in base_rows}
    if len(base_rows) != 60 or set(by_id) != train_ids:
        raise ValueError("Staging export manifest must contain exactly the frozen 60 training patients")
    sources = {row["case_id"]: row for row in source_rows}
    held_out_fold = {case: fold["fold"] for fold in split["folds"] for case in fold["held_out"]}
    print("Phase 1: frozen list and task ID preflight passed; verifying 60 unique CT/mask pairs", flush=True)
    for number, case in enumerate(sorted(train_ids), 1):
        row, source = by_id[case], sources[case]
        expected_fields = {
            "source_split": source["source_split"], "qc_flag": source["qc_flag"],
            "source_image": source["image"], "source_mask": source["majority_mask"],
            "export_image": str(base_dir / "imagesTr" / f"{case}_0000.nii.gz"),
            "export_mask": str(base_dir / "labelsTr" / f"{case}.nii.gz"),
            "oof_held_out_fold": str(held_out_fold[case]),
        }
        if any(row[key] != value for key, value in expected_fields.items()):
            raise ValueError(f"Staging provenance mismatch: {case}")
        for kind in ("image", "mask"):
            # Hash each unique source and staging file once, not four times for its links.
            paths = {Path(row[f"source_{kind}"]).resolve(), Path(row[f"export_{kind}"]).resolve()}
            if any(sha256(path) != row[f"{kind}_sha256"] for path in paths):
                raise ValueError(f"Source/staging hash mismatch: {case}/{kind}")
        assert_same_geometry(Path(row["export_image"]), Path(row["export_mask"]))
        if number % 10 == 0:
            print(f"  verified {number}/60 unique pairs (hashes + geometry)", flush=True)
    print("Phase 1 self-check passed: frozen split, source/staging hashes and geometry", flush=True)
    return split, options, base_dir, by_id


def task_directory(settings: dict, task: dict) -> Path:
    return Path(settings["nnunet"]["raw_root"]) / (
        f"Dataset{task['dataset_id']:03d}_{task['dataset_name']}"
    )


def verify_task(dataset_dir: Path, fold: dict, base_rows: dict[str, dict[str, str]]) -> None:
    train_ids = set(fold["train"])
    check_directory_names(dataset_dir, {"imagesTr", "labelsTr", "dataset.json"})
    check_directory_names(dataset_dir / "imagesTr", {f"{case}_0000.nii.gz" for case in train_ids})
    check_directory_names(dataset_dir / "labelsTr", {f"{case}.nii.gz" for case in train_ids})
    if json.loads((dataset_dir / "dataset.json").read_text(encoding="utf-8")) != dataset_metadata(48):
        raise ValueError(f"Incorrect single-channel CT metadata: {dataset_dir}")
    for case in sorted(train_ids):
        row = base_rows[case]
        check_link(dataset_dir / "imagesTr" / f"{case}_0000.nii.gz", Path(row["export_image"]))
        check_link(dataset_dir / "labelsTr" / f"{case}.nii.gz", Path(row["export_mask"]))
    print(f"  Fold {fold['fold']} self-check passed: 48 CT/mask pairs; 12 held-out excluded", flush=True)


def manifest_contents(settings: dict, split: dict, options: dict,
                      base_dir: Path, base_rows: dict[str, dict[str, str]]) -> tuple[str, str]:
    folds = {fold["fold"]: fold for fold in split["folds"]}
    output = {
        "schema_version": 1,
        "split_manifest": settings["split_manifest"],
        "split_sha256": options["split_sha256"],
        "staging_dataset": str(base_dir),
        "staging_export_manifest": "data/manifests/curvas_nnunet_export.csv",
        "staging_export_sha256": sha256(Path("data/manifests/curvas_nnunet_export.csv")),
        "link_mode": options["link_mode"],
        "training_fold": options["training_fold"],
        "tasks": [],
    }
    csv_rows = []
    for task in sorted(options["datasets"], key=lambda item: item["fold"]):
        fold, dataset_dir = folds[task["fold"]], task_directory(settings, task)
        output["tasks"].append({
            **task, "dataset_dir": str(dataset_dir), "numTraining": 48, "numHeldOut": 12,
            "train": fold["train"], "held_out": fold["held_out"],
            # Inference image references only: no held-out masks enter a raw task.
            "held_out_inputs": [
                {"case_id": case, "image": base_rows[case]["export_image"]}
                for case in fold["held_out"]
            ],
        })
        for case in fold["train"]:
            row = base_rows[case]
            csv_rows.append({
                "fold": task["fold"], "dataset_id": task["dataset_id"], "case_id": case,
                "source_split": row["source_split"], "qc_flag": row["qc_flag"],
                "source_image": row["source_image"], "source_mask": row["source_mask"],
                "staging_image": row["export_image"], "staging_mask": row["export_mask"],
                "task_image": str(dataset_dir / "imagesTr" / f"{case}_0000.nii.gz"),
                "task_mask": str(dataset_dir / "labelsTr" / f"{case}.nii.gz"),
                "image_sha256": row["image_sha256"], "mask_sha256": row["mask_sha256"],
                "split_sha256": options["split_sha256"],
            })
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(csv_rows[0]))
    writer.writeheader()
    writer.writerows(csv_rows)
    return json.dumps(output, ensure_ascii=False, indent=2) + "\n", buffer.getvalue()


def verify_manifest(path: Path, expected: str) -> None:
    # Preserve CSV CRLF when comparing the reproducible serialization.
    with path.open(encoding="utf-8", newline="") as stream:
        actual = stream.read()
    if actual != expected:
        raise ValueError(f"Task manifest differs from frozen inputs: {path}")


def write_manifest(path: Path, content: str) -> None:
    if path.exists():
        verify_manifest(path, content)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        stream.write(content)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/data.yaml"))
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--verify-nnunet", action="store_true",
                        help="Additionally run native nnU-Net integrity checks on all five tasks")
    parser.add_argument("--num-processes", type=int, default=4)
    args = parser.parse_args()
    if args.num_processes < 1:
        parser.error("--num-processes must be positive")
    settings = yaml.safe_load(args.config.read_text(encoding="utf-8"))["curvas"]
    split, options, base_dir, base_rows = load_inputs(settings)
    folds = {fold["fold"]: fold for fold in split["folds"]}
    task_json, task_csv = manifest_contents(settings, split, options, base_dir, base_rows)
    # Check all existing metadata before creating any new links.
    for path, content in ((Path(options["manifest"]), task_json),
                          (Path(options["export_manifest"]), task_csv)):
        if path.exists():
            verify_manifest(path, content)
    for task in options["datasets"]:
        preflight_task(task_directory(settings, task), folds[task["fold"]], base_rows)
    for task in sorted(options["datasets"], key=lambda item: item["fold"]):
        dataset_dir, fold = task_directory(settings, task), folds[task["fold"]]
        if not args.verify_only:
            dataset_dir.mkdir(parents=True, exist_ok=True)
            ensure_text(dataset_dir / "dataset.json", json.dumps(dataset_metadata(48), indent=2) + "\n")
            for case in fold["train"]:
                row = base_rows[case]
                ensure_link(dataset_dir / "imagesTr" / f"{case}_0000.nii.gz", Path(row["export_image"]))
                ensure_link(dataset_dir / "labelsTr" / f"{case}.nii.gz", Path(row["export_mask"]))
        verify_task(dataset_dir, fold, base_rows)
    if not args.verify_only:
        write_manifest(Path(options["manifest"]), task_json)
        write_manifest(Path(options["export_manifest"]), task_csv)
    verify_manifest(Path(options["manifest"]), task_json)
    verify_manifest(Path(options["export_manifest"]), task_csv)
    validate_fold_protocol(split)
    print("Phase 2 self-check passed: five 48-patient tasks; each patient trains 4x and is held out 1x", flush=True)
    print("Phase 3 self-check passed: task JSON and 240-row export CSV match frozen inputs", flush=True)
    if args.verify_nnunet:
        from nnunetv2.experiment_planning.verify_dataset_integrity import verify_dataset_integrity

        for task in sorted(options["datasets"], key=lambda item: item["fold"]):
            dataset_dir = task_directory(settings, task)
            print(f"Native nnU-Net integrity check: {dataset_dir}", flush=True)
            verify_dataset_integrity(str(dataset_dir), num_processes=args.num_processes)
            print(f"Native self-check passed: fold {task['fold']} / dataset {task['dataset_id']}", flush=True)
    print("Done. No fingerprinting, planning, preprocessing or training performed.", flush=True)


if __name__ == "__main__":
    main()
