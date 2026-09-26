"""Export only frozen CURVAS training patients to a nnU-Net v2 raw dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import yaml


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_exact(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        source_hash = sha256(source)
        if sha256(destination) != source_hash:
            raise ValueError(f"Existing export differs from its source: {destination}")
        return source_hash

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    try:
        with source.open("rb") as input_stream, os.fdopen(descriptor, "wb") as output_stream:
            for block in iter(lambda: input_stream.read(8 * 1024 * 1024), b""):
                output_stream.write(block)
                digest.update(block)
        if sha256(temporary) != digest.hexdigest():
            raise ValueError(f"Copy verification failed: {destination}")
        temporary.chmod(0o644)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return digest.hexdigest()


def geometry(path: Path) -> tuple[tuple[int, ...], np.ndarray, np.ndarray, np.ndarray]:
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    reader.ReadImageInformation()
    return (
        reader.GetSize(),
        np.asarray(reader.GetSpacing()),
        np.asarray(reader.GetOrigin()),
        np.asarray(reader.GetDirection()),
    )


def assert_same_geometry(image: Path, mask: Path) -> None:
    image_geometry, mask_geometry = geometry(image), geometry(mask)
    if image_geometry[0] != mask_geometry[0] or any(
        not np.allclose(left, right, rtol=1e-5, atol=1e-3)
        for left, right in zip(image_geometry[1:], mask_geometry[1:])
    ):
        raise ValueError(f"CT and mask geometry differ: {image} / {mask}")


def ensure_text(path: Path, content: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise ValueError(f"Existing export metadata differs: {path}")
    else:
        path.write_text(content, encoding="utf-8")


def expected_export(settings: dict) -> tuple[dict, list[dict[str, str]], Path, Path]:
    split_path = Path(settings["split_manifest"])
    split = json.loads(split_path.read_text(encoding="utf-8"))
    source_manifest = Path(settings["manifest"])
    dispositions = Path(settings["dispositions"])
    if split["input_sha256"] != {
        "curvas_oof_label_manifest.csv": sha256(source_manifest),
        "curvas_case_dispositions.csv": sha256(dispositions),
    }:
        raise ValueError("Frozen split no longer matches its source manifests")
    if len(split["splits"]["train"]) != 60:
        raise ValueError("Expected exactly 60 training patients")
    source_rows = read_csv(source_manifest)
    by_id = {row["case_id"]: row for row in source_rows}
    if len(by_id) != len(source_rows) or set(split["splits"]["train"]) - set(by_id):
        raise ValueError("Training IDs are missing or duplicated in source manifest")
    nnunet = settings["nnunet"]
    dataset_dir = Path(nnunet["raw_root"]) / (
        f"Dataset{int(nnunet['dataset_id']):03d}_{nnunet['dataset_name']}"
    )
    export_manifest = Path("data/manifests/curvas_nnunet_export.csv")
    return split, source_rows, dataset_dir, export_manifest


def verify_export(settings: dict, split: dict, source_rows: list[dict[str, str]], dataset_dir: Path,
                  export_manifest: Path) -> None:
    train_ids = set(split["splits"]["train"])
    image_dir, label_dir = dataset_dir / "imagesTr", dataset_dir / "labelsTr"
    images = {path.name for path in image_dir.glob("*.nii.gz")}
    labels = {path.name for path in label_dir.glob("*.nii.gz")}
    if images != {f"{case_id}_0000.nii.gz" for case_id in train_ids}:
        raise ValueError("Exported CT IDs differ from the frozen 60-patient training list")
    if labels != {f"{case_id}.nii.gz" for case_id in train_ids}:
        raise ValueError("Exported mask IDs differ from the frozen 60-patient training list")
    if (dataset_dir / "imagesTs").exists():
        raise ValueError("imagesTs must not be present in this training-only export")
    expected_json = {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "pancreas": 1},
        "numTraining": 60,
        "file_ending": ".nii.gz",
    }
    if json.loads((dataset_dir / "dataset.json").read_text(encoding="utf-8")) != expected_json:
        raise ValueError("Incorrect nnU-Net dataset.json")
    by_id = {row["case_id"]: row for row in source_rows}
    rows = read_csv(export_manifest)
    if len(rows) != 60 or {row["case_id"] for row in rows} != train_ids:
        raise ValueError("Export manifest does not contain the 60 training cases")
    for row in rows:
        case_id = row["case_id"]
        source = by_id[case_id]
        image, mask = image_dir / f"{case_id}_0000.nii.gz", label_dir / f"{case_id}.nii.gz"
        if row["source_image"] != source["image"] or row["source_mask"] != source["majority_mask"]:
            raise ValueError(f"Export manifest source mismatch: {case_id}")
        if row["image_sha256"] != sha256(image) or row["mask_sha256"] != sha256(mask):
            raise ValueError(f"Export checksum mismatch: {case_id}")
        if row["image_sha256"] != sha256(Path(source["image"])) or row["mask_sha256"] != sha256(Path(source["majority_mask"])):
            raise ValueError(f"Export differs from source: {case_id}")
        assert_same_geometry(image, mask)
    print(f"Verified {len(rows)} CT/mask pairs, source hashes, geometry, and single-channel CT metadata")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/data.yaml"))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    settings = yaml.safe_load(args.config.read_text(encoding="utf-8"))["curvas"]
    split, source_rows, dataset_dir, export_manifest = expected_export(settings)
    if args.verify_only:
        verify_export(settings, split, source_rows, dataset_dir, export_manifest)
        return

    by_id = {row["case_id"]: row for row in source_rows}
    held_out_fold = {
        case_id: fold["fold"]
        for fold in split["folds"]
        for case_id in fold["held_out"]
    }
    if set(held_out_fold) != set(split["splits"]["train"]):
        raise ValueError("Fold assignments do not cover exactly the training patients")
    image_dir, label_dir = dataset_dir / "imagesTr", dataset_dir / "labelsTr"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    output_rows = []
    for number, case_id in enumerate(split["splits"]["train"], 1):
        source = by_id[case_id]
        source_image, source_mask = Path(source["image"]), Path(source["majority_mask"])
        assert_same_geometry(source_image, source_mask)
        target_image = image_dir / f"{case_id}_0000.nii.gz"
        target_mask = label_dir / f"{case_id}.nii.gz"
        image_hash = copy_exact(source_image, target_image)
        mask_hash = copy_exact(source_mask, target_mask)
        output_rows.append({
            "case_id": case_id,
            "source_split": source["source_split"],
            "oof_held_out_fold": held_out_fold[case_id],
            "qc_flag": source["qc_flag"],
            "source_image": str(source_image),
            "source_mask": str(source_mask),
            "export_image": str(target_image),
            "export_mask": str(target_mask),
            "image_sha256": image_hash,
            "mask_sha256": mask_hash,
        })
        print(f"[{number:02d}/60] {case_id} exported", flush=True)
    metadata = {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "pancreas": 1},
        "numTraining": 60,
        "file_ending": ".nii.gz",
    }
    ensure_text(dataset_dir / "dataset.json", json.dumps(metadata, indent=2) + "\n")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(output_rows[0]))
    writer.writeheader()
    writer.writerows(output_rows)
    ensure_text(export_manifest, buffer.getvalue())
    print(f"Exported {len(output_rows)} cases to {dataset_dir}; run --verify-only for the independent check")


if __name__ == "__main__":
    main()
