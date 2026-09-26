"""Freeze patient-level CURVAS splits and five OOF hold-out folds."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import yaml


SPLIT_NAMES = ("train", "calibration", "test")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_split(settings: dict, manifest_path: Path, decisions_path: Path) -> dict:
    rows = read_csv(manifest_path)
    decisions = read_csv(decisions_path)
    by_id = {row["case_id"]: row for row in rows}
    if len(rows) != 90 or len(by_id) != 90:
        raise ValueError("Expected 90 unique CURVAS patients")
    if any(not Path(row["image"]).is_file() or not Path(row["majority_mask"]).is_file() for row in rows):
        raise ValueError("Missing CT or majority-vote mask")
    decision_by_id = {row["case_id"]: row for row in decisions}
    if len(decision_by_id) != len(decisions):
        raise ValueError("Duplicate case disposition")
    if any(case_id not in by_id for case_id in decision_by_id):
        raise ValueError("Case disposition refers to an unknown patient")
    for case_id, row in by_id.items():
        if row["qc_flag"] != "PASS_AUTOMATED":
            decision = decision_by_id.get(case_id, {}).get("decision")
            if decision != "RETAIN_CT_SPACE_MAJORITY":
                raise ValueError(f"Unresolved QC case without retain decision: {case_id}")
    if not {"UKCHLL007", "UKCHLL008", "UKCHLL082"}.issubset(decision_by_id):
        raise ValueError("Missing explicit user retention decision")

    quotas = settings["source_quotas"]
    counts = settings["counts"]
    if set(quotas) != {row["source_split"] for row in rows}:
        raise ValueError("Source quota keys do not match original archives")
    if any(set(quota) != set(SPLIT_NAMES) for quota in quotas.values()):
        raise ValueError("Every source must have train/calibration/test quotas")
    for source, quota in quotas.items():
        source_count = sum(row["source_split"] == source for row in rows)
        if sum(quota.values()) != source_count:
            raise ValueError(f"Quotas for {source} do not sum to {source_count}")
    if any(sum(quotas[source][name] for source in quotas) != counts[name] for name in SPLIT_NAMES):
        raise ValueError("Source quotas do not match target split counts")

    rng = random.Random(int(settings["seed"]))
    splits: dict[str, list[str]] = {name: [] for name in SPLIT_NAMES}
    train_by_source: dict[str, list[str]] = {}
    for source in sorted(quotas):
        ids = sorted(row["case_id"] for row in rows if row["source_split"] == source)
        rng.shuffle(ids)
        offset = 0
        for name in SPLIT_NAMES:
            selection = ids[offset : offset + quotas[source][name]]
            splits[name].extend(selection)
            if name == "train":
                train_by_source[source] = selection
            offset += quotas[source][name]
    for ids in splits.values():
        ids.sort()

    fold_count = int(settings["oof_folds"])
    if counts["train"] % fold_count:
        raise ValueError("Train count must divide evenly into OOF folds")
    fold_size = counts["train"] // fold_count
    held_out: list[list[str]] = [[] for _ in range(fold_count)]
    source_counts = [Counter() for _ in range(fold_count)]
    for source in sorted(train_by_source, key=lambda name: (-len(train_by_source[name]), name)):
        for case_id in train_by_source[source]:
            eligible = [fold for fold in range(fold_count) if len(held_out[fold]) < fold_size]
            fold = min(eligible, key=lambda index: (source_counts[index][source], len(held_out[index]), index))
            held_out[fold].append(case_id)
            source_counts[fold][source] += 1
    folds = [
        {
            "fold": index,
            "held_out": sorted(held_out[index]),
            "train": sorted(set(splits["train"]) - set(held_out[index])),
        }
        for index in range(fold_count)
    ]

    frozen = {
        "schema_version": 1,
        "seed": int(settings["seed"]),
        "counts": counts,
        "source_quotas": quotas,
        "input_sha256": {
            "curvas_oof_label_manifest.csv": sha256(manifest_path),
            "curvas_case_dispositions.csv": sha256(decisions_path),
        },
        "splits": splits,
        "folds": folds,
    }
    validate_split(frozen, by_id)
    return frozen


def validate_split(frozen: dict, by_id: dict[str, dict[str, str]]) -> None:
    splits = frozen["splits"]
    if set(splits) != set(SPLIT_NAMES):
        raise ValueError("Incorrect split names")
    all_split_ids = [case_id for name in SPLIT_NAMES for case_id in splits[name]]
    if len(all_split_ids) != len(by_id) or set(all_split_ids) != set(by_id):
        raise ValueError("Patient split is incomplete or overlapping")
    for name in SPLIT_NAMES:
        if len(splits[name]) != frozen["counts"][name]:
            raise ValueError(f"Incorrect {name} patient count")
    for source, quota in frozen["source_quotas"].items():
        for name, expected in quota.items():
            actual = sum(by_id[case_id]["source_split"] == source for case_id in splits[name])
            if actual != expected:
                raise ValueError(f"{source}/{name}: expected {expected}, got {actual}")
    train_ids = set(splits["train"])
    held_out_ids = []
    for fold in frozen["folds"]:
        held_out = set(fold["held_out"])
        fold_train = set(fold["train"])
        if len(held_out) != 12 or len(fold_train) != 48 or held_out & fold_train:
            raise ValueError(f"Invalid OOF fold {fold['fold']}")
        if held_out | fold_train != train_ids:
            raise ValueError(f"OOF fold {fold['fold']} has an incorrect patient universe")
        held_out_ids.extend(held_out)
    if len(held_out_ids) != len(train_ids) or set(held_out_ids) != train_ids:
        raise ValueError("Each training patient must be held out exactly once")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/data.yaml"))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    settings = yaml.safe_load(args.config.read_text(encoding="utf-8"))["curvas"]
    manifest_path = Path(settings["manifest"])
    decisions_path = Path(settings["dispositions"])
    output_path = Path(settings["split_manifest"])
    frozen = build_split(settings, manifest_path, decisions_path)
    if output_path.exists():
        actual = json.loads(output_path.read_text(encoding="utf-8"))
        if actual != frozen:
            raise ValueError(f"Existing frozen split differs from inputs/config: {output_path}")
        print(f"Verified frozen split: {output_path}")
    elif args.verify_only:
        raise FileNotFoundError(output_path)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(frozen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Created frozen split: {output_path}")
    print(f"Counts: { {name: len(frozen['splits'][name]) for name in SPLIT_NAMES} }")
    print(f"OOF held-out sizes: {[len(fold['held_out']) for fold in frozen['folds']]}")
    print(f"SHA-256: {sha256(output_path)}")


if __name__ == "__main__":
    main()
