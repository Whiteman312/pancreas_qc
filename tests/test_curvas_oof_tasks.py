"""Small synthetic regression tests; no medical data or nnU-Net training needed."""

from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_curvas_oof_tasks import (  # noqa: E402
    check_link,
    dataset_metadata,
    ensure_link,
    preflight_task,
    validate_fold_protocol,
    verify_manifest,
    verify_task,
    write_manifest,
)


def synthetic_split() -> dict:
    train = [f"CASE{number:03d}" for number in range(60)]
    return {
        "counts": {"train": 60, "calibration": 10, "test": 20},
        "splits": {
            "train": train,
            "calibration": [f"CAL{number:03d}" for number in range(10)],
            "test": [f"TEST{number:03d}" for number in range(20)],
        },
        "folds": [
            {"fold": index, "held_out": train[index * 12:(index + 1) * 12],
             "train": train[:index * 12] + train[(index + 1) * 12:]}
            for index in range(5)
        ],
    }


class FoldProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.split = synthetic_split()

    def test_valid_protocol(self) -> None:
        validate_fold_protocol(self.split)

    def test_reject_held_out_in_training(self) -> None:
        self.split["folds"][0]["train"][0] = self.split["folds"][0]["held_out"][0]
        with self.assertRaises(ValueError):
            validate_fold_protocol(self.split)

    def test_reject_calibration_or_test_in_training(self) -> None:
        for partition in ("calibration", "test"):
            split = copy.deepcopy(self.split)
            split["folds"][0]["train"][0] = split["splits"][partition][0]
            with self.subTest(partition=partition), self.assertRaises(ValueError):
                validate_fold_protocol(split)

    def test_reject_duplicate_training_patient(self) -> None:
        self.split["folds"][0]["train"][0] = self.split["folds"][0]["train"][1]
        with self.assertRaises(ValueError):
            validate_fold_protocol(self.split)

    def test_reject_duplicate_fold_index(self) -> None:
        self.split["folds"][1]["fold"] = 0
        with self.assertRaises(ValueError):
            validate_fold_protocol(self.split)

    def test_reject_repeated_holdout_fold(self) -> None:
        duplicate = copy.deepcopy(self.split["folds"][0])
        duplicate["fold"] = 1
        self.split["folds"][1] = duplicate
        with self.assertRaises(ValueError):
            validate_fold_protocol(self.split)

    def test_reject_overlapping_partitions(self) -> None:
        self.split["splits"]["calibration"][0] = self.split["splits"]["train"][0]
        with self.assertRaises(ValueError):
            validate_fold_protocol(self.split)


class RawTaskTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="curvas-oof-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.fold = synthetic_split()["folds"][0]
        self.base_rows = {}
        for case in synthetic_split()["splits"]["train"]:
            image, mask = self.root / "base" / "imagesTr" / f"{case}_0000.nii.gz", self.root / "base" / "labelsTr" / f"{case}.nii.gz"
            image.parent.mkdir(parents=True, exist_ok=True)
            mask.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(b"synthetic CT; directory tests only")
            mask.write_bytes(b"synthetic mask; directory tests only")
            self.base_rows[case] = {"export_image": str(image), "export_mask": str(mask)}
        self.task = self.root / "Dataset511_CURVASPancreasOOF_Fold0"
        self.task.mkdir()
        (self.task / "dataset.json").write_text(json.dumps(dataset_metadata(48), indent=2) + "\n", encoding="utf-8")
        for case in self.fold["train"]:
            row = self.base_rows[case]
            ensure_link(self.task / "imagesTr" / f"{case}_0000.nii.gz", Path(row["export_image"]))
            ensure_link(self.task / "labelsTr" / f"{case}.nii.gz", Path(row["export_mask"]))

    def test_valid_relative_links_and_idempotent_creation(self) -> None:
        preflight_task(self.task, self.fold, self.base_rows)
        verify_task(self.task, self.fold, self.base_rows)
        case = self.fold["train"][0]
        target = self.task / "imagesTr" / f"{case}_0000.nii.gz"
        source = Path(self.base_rows[case]["export_image"])
        before = target.lstat().st_ino
        ensure_link(target, source)
        self.assertEqual(before, target.lstat().st_ino)
        self.assertFalse(os.path.isabs(os.readlink(target)))
        self.assertEqual(target.read_bytes(), source.read_bytes())

    def test_reject_held_out_image_in_task(self) -> None:
        case = self.fold["held_out"][0]
        ensure_link(self.task / "imagesTr" / f"{case}_0000.nii.gz", Path(self.base_rows[case]["export_image"]))
        with self.assertRaises(ValueError):
            preflight_task(self.task, self.fold, self.base_rows)
        with self.assertRaises(ValueError):
            verify_task(self.task, self.fold, self.base_rows)

    def test_reject_wrong_link_target(self) -> None:
        case, other = self.fold["train"][:2]
        target = self.task / "imagesTr" / f"{case}_0000.nii.gz"
        target.unlink()
        target.symlink_to(os.path.relpath(self.base_rows[other]["export_image"], start=target.parent))
        with self.assertRaises(ValueError):
            ensure_link(target, Path(self.base_rows[case]["export_image"]))

    def test_reject_broken_link_without_overwrite(self) -> None:
        target = self.root / "broken.nii.gz"
        target.symlink_to("missing.nii.gz")
        source = Path(next(iter(self.base_rows.values()))["export_image"])
        with self.assertRaises(ValueError):
            ensure_link(target, source)
        self.assertEqual(os.readlink(target), "missing.nii.gz")

    def test_reject_absolute_link(self) -> None:
        source = Path(next(iter(self.base_rows.values()))["export_image"])
        target = self.root / "absolute.nii.gz"
        target.symlink_to(source)
        with self.assertRaises(ValueError):
            check_link(target, source)

    def test_preserve_existing_regular_file(self) -> None:
        target = self.root / "existing.nii.gz"
        target.write_bytes(b"do not overwrite")
        source = Path(next(iter(self.base_rows.values()))["export_image"])
        with self.assertRaises(ValueError):
            ensure_link(target, source)
        self.assertEqual(target.read_bytes(), b"do not overwrite")

    def test_reject_wrong_metadata(self) -> None:
        metadata = dataset_metadata(48)
        metadata["channel_names"] = {"0": "MR"}
        (self.task / "dataset.json").write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaises(ValueError):
            preflight_task(self.task, self.fold, self.base_rows)
        with self.assertRaises(ValueError):
            verify_task(self.task, self.fold, self.base_rows)

    def test_reject_images_ts(self) -> None:
        (self.task / "imagesTs").mkdir()
        with self.assertRaises(ValueError):
            verify_task(self.task, self.fold, self.base_rows)

    def test_reject_symlinked_task_root(self) -> None:
        alias = self.root / "task-alias"
        alias.symlink_to(self.task, target_is_directory=True)
        with self.assertRaises(ValueError):
            preflight_task(alias, self.fold, self.base_rows)

    def test_csv_crlf_roundtrip_and_conflict_rejection(self) -> None:
        path, content = self.root / "manifest.csv", "fold,case_id\r\n0,CASE012\r\n"
        write_manifest(path, content)
        write_manifest(path, content)
        verify_manifest(path, content)
        with self.assertRaises(ValueError):
            write_manifest(path, "fold,case_id\r\n0,CASE013\r\n")
        self.assertEqual(path.read_bytes(), content.encode("utf-8"))


if __name__ == "__main__":
    unittest.main()
