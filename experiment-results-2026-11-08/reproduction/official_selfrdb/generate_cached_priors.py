"""Generate official SelfRDB prior caches for the audited patient split."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    official_root = root / "official_selfrdb"
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(official_root))

    from main import BridgeRunner
    from datasets import CachedCTMRIDataset

    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    runner = BridgeRunner.load_from_checkpoint(
        args.checkpoint,
        map_location=device,
    )
    runner.to(device)
    runner.eval()

    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        target = out_dir / f"official_{split}.npy"
        names_target = out_dir / f"official_{split}_names.json"
        if target.exists() and not args.force:
            print(f"skip existing {target}", flush=True)
            continue
        dataset = CachedCTMRIDataset(
            str(root),
            target_modality="MR",
            source_modality="CT",
            stage=split,
            image_size=128,
            norm=False,
            padding=False,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            persistent_workers=True,
            pin_memory=True,
        )
        predictions = []
        start = time.time()
        with torch.inference_mode():
            for index, (_, ct, _) in enumerate(loader):
                prediction = runner.diffusion.sample_x0(
                    ct.to(device, non_blocking=True),
                    runner.generator,
                )
                predictions.append(prediction.cpu().numpy())
                if (index + 1) % 50 == 0:
                    done = min((index + 1) * args.batch_size, len(dataset))
                    print(
                        f"{split}: {done}/{len(dataset)} "
                        f"elapsed={time.time() - start:.0f}s",
                        flush=True,
                    )
        array = np.concatenate(predictions, axis=0).astype(np.float32)
        np.save(target, array)
        names_target.write_text(
            json.dumps(dataset.names, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            f"saved {target} shape={array.shape} "
            f"range=({array.min():.3f},{array.max():.3f}) "
            f"elapsed={time.time() - start:.0f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
