"""Audit patient splits, cached arrays, metrics, and checkpoints.

This script is read-only with respect to experiment artifacts. It writes an
audit report into a separate output directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path):
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def patient_from_name(name: str) -> str:
    return str(name).replace("\\", "/").split("/", 1)[0]


def audit(root: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, object] = {
        "root": str(root),
        "checks": [],
        "files": [],
    }

    def check(name: str, passed: bool, details: object) -> None:
        result["checks"].append(
            {"name": name, "passed": bool(passed), "details": details}
        )

    split_candidates = [
        root / "checkpoints/selfrdb/patient_split.json",
        root / "other_model/SelfRDB/patient_split.json",
    ]
    split_path = next((p for p in split_candidates if p.exists()), None)
    if split_path is None:
        check("split_file_exists", False, [str(p) for p in split_candidates])
        split = {}
    else:
        check("split_file_exists", True, str(split_path))
        split = load_json(split_path)

    split_sets = {
        key: set(str(v) for v in split.get(key, []))
        for key in ("train", "val", "test")
    }
    split_counts = {key: len(value) for key, value in split_sets.items()}
    check(
        "split_counts",
        split_counts == {"train": 144, "val": 18, "test": 18},
        split_counts,
    )
    overlaps = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sorted(split_sets[left] & split_sets[right])
        overlaps[f"{left}_{right}"] = overlap
    check(
        "patient_split_disjoint",
        not any(overlaps.values()),
        overlaps,
    )
    union = set().union(*split_sets.values()) if split_sets else set()
    check("patient_union_count", len(union) == 180, len(union))

    cache_dir = root / "experiments/controls/cache"
    cache_names = {
        split_name: cache_dir / f"names_{split_name}.json"
        for split_name in ("train", "val", "test")
    }
    cache_arrays = {
        ("ct", "train"): cache_dir / "ct_train.npy",
        ("ct", "val"): cache_dir / "ct_val.npy",
        ("ct", "test"): cache_dir / "ct_test.npy",
        ("mr", "train"): cache_dir / "mr_train.npy",
        ("mr", "val"): cache_dir / "mr_val.npy",
        ("mr", "test"): cache_dir / "mr_test.npy",
        ("pix2pix_prior", "train"): cache_dir / "pix2pix_prior_train.npy",
        ("pix2pix_prior", "test"): cache_dir / "pix2pix_prior_test.npy",
        ("coarse_2x1", "train"): cache_dir / "coarse_2x1_train.npy",
        ("detreg", "train"): root / "experiments/controls/detreg_coarse_train.npy",
        ("detreg", "val"): root / "experiments/controls/detreg_coarse_val.npy",
        ("detreg", "test"): root / "experiments/controls/detreg_coarse_test.npy",
        ("bridge_10x2", "test"): root / "experiments/controls/bridge_eval_coarse_test.npy",
    }

    names_by_split: dict[str, list[str]] = {}
    for split_name, path in cache_names.items():
        if not path.exists():
            check(f"names_{split_name}_exists", False, str(path))
            continue
        names = [str(v) for v in load_json(path)]
        names_by_split[split_name] = names
        check(f"names_{split_name}_exists", True, str(path))
        check(f"names_{split_name}_nonempty", len(names) > 0, len(names))
        check(
            f"names_{split_name}_unique",
            len(names) == len(set(names)),
            {"count": len(names), "unique": len(set(names))},
        )
        name_patients = {patient_from_name(name) for name in names}
        expected = split_sets.get(split_name, set())
        missing = sorted(expected - name_patients)
        extra = sorted(name_patients - expected)
        check(
            f"names_{split_name}_patients_match_split",
            not missing and not extra,
            {"missing": missing, "extra": extra},
        )

    array_shapes = {}
    for (source, split_name), path in cache_arrays.items():
        if not path.exists():
            check(f"cache_{source}_{split_name}_exists", False, str(path))
            continue
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        array_shapes[f"{source}_{split_name}"] = list(array.shape)
        check(f"cache_{source}_{split_name}_exists", True, str(path))
        check(
            f"cache_{source}_{split_name}_finite",
            bool(np.isfinite(np.asarray(array[: min(32, len(array))])).all()),
            "checked first 32 slices",
        )
        expected_n = len(names_by_split.get(split_name, []))
        check(
            f"cache_{source}_{split_name}_row_count",
            expected_n == len(array),
            {"array": len(array), "names": expected_n},
        )
        if array.ndim == 4:
            check(
                f"cache_{source}_{split_name}_single_channel",
                array.shape[1] == 1,
                list(array.shape),
            )

    for split_name in ("train", "val", "test"):
        if (
            ("ct", split_name) in array_shapes
            and ("mr", split_name) in array_shapes
        ):
            check(
                f"ct_mr_{split_name}_shape_match",
                array_shapes[("ct", split_name)] == array_shapes[("mr", split_name)],
                {
                    "ct": array_shapes[("ct", split_name)],
                    "mr": array_shapes[("mr", split_name)],
                },
            )

    metrics_path = (
        root
        / "other_model/paper/source_data/unified10x2/"
        "metrics_patient_unified_10x2.csv"
    )
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8-sig", newline="") as handle:
            metric_rows = list(csv.DictReader(handle))
        metric_patients = {row["patient"] for row in metric_rows}
        check("metrics_patient_file_exists", True, str(metrics_path))
        check("metrics_patient_count", len(metric_rows) == 18, len(metric_rows))
        check(
            "metrics_patients_match_test_split",
            metric_patients == split_sets.get("test", set()),
            {
                "missing": sorted(split_sets.get("test", set()) - metric_patients),
                "extra": sorted(metric_patients - split_sets.get("test", set())),
            },
        )
    else:
        check("metrics_patient_file_exists", False, str(metrics_path))

    critical_files = [
        split_path,
        *cache_names.values(),
        *cache_arrays.values(),
        metrics_path,
        root / "other_model/SelfRDB/selfrdb_best.pt",
        root / "experiments/controls/E1_detreg_seed42/best.pt",
        root / "experiments/controls/E2_capct_seed42/best.pt",
        root / "experiments/controls/E3_bridgerefine_seed42/best.pt",
        root / "experiments/controls/E4_ct_only_seed42/best.pt",
        root / "experiments/gan_prior/last.pt",
        root / "experiments/gan_prior/seed123/last.pt",
        root / "experiments/gan_prior/seed2026/last.pt",
        root / "experiments/coarse_2x1/last.pt",
        root / "experiments/coarse_2x1/seed123/last.pt",
        root / "experiments/coarse_2x1/seed2026/last.pt",
    ]
    seen = set()
    for path in critical_files:
        if path is None:
            continue
        path = Path(path)
        if path in seen:
            continue
        seen.add(path)
        if path.exists():
            result["files"].append(
                {
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
        else:
            result["files"].append(
                {"path": str(path), "size_bytes": None, "sha256": None}
            )

    result["summary"] = {
        "total_checks": len(result["checks"]),
        "passed": sum(1 for item in result["checks"] if item["passed"]),
        "failed": sum(1 for item in result["checks"] if not item["passed"]),
        "split_counts": split_counts,
        "names_counts": {
            key: len(value) for key, value in names_by_split.items()
        },
        "array_shapes": array_shapes,
    }
    (out_dir / "audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with (out_dir / "file_hashes.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["path", "size_bytes", "sha256"]
        )
        writer.writeheader()
        writer.writerows(result["files"])

    lines = [
        "# Stage 0 audit",
        "",
        f"- Root: `{root}`",
        f"- Checks: {result['summary']['passed']} passed, "
        f"{result['summary']['failed']} failed",
        f"- Split counts: `{split_counts}`",
        f"- Cached name counts: `{result['summary']['names_counts']}`",
        "",
        "## Failed checks",
        "",
    ]
    failed = [item for item in result["checks"] if not item["passed"]]
    if not failed:
        lines.append("None.")
    else:
        for item in failed:
            lines.append(
                f"- **{item['name']}**: `{json.dumps(item['details'], ensure_ascii=False)}`"
            )
    lines.extend(["", "## All checks", "", "| Check | Result |", "|---|---|"])
    for item in result["checks"]:
        lines.append(f"| {item['name']} | {'PASS' if item['passed'] else 'FAIL'} |")
    (out_dir / "audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.root.resolve(), args.out.resolve())
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
