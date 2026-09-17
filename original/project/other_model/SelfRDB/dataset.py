# -*- coding: utf-8 -*-
"""
Dataset for paired CT-MRI brain slices from preprocessed NIfTI volumes.

Expected directory structure:
    data_slices/
        {patient_id}/
            ct/  0000.png, 0001.png, ...
            mr/  0000.png, 0001.png, ...

Splits by patient to prevent data leakage.
"""
import random
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def list_images(folder):
    folder = Path(folder)
    if not folder.exists():
        return {}
    return {
        path.stem: path
        for path in sorted(folder.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }


class PairedCTMRIDataset(Dataset):
    """Loads paired CT-MRI slices from multiple patient directories."""

    def __init__(self, data_root, image_size=128, augment=False, patient_ids=None):
        data_root = Path(data_root)
        all_patients = sorted(
            d for d in data_root.iterdir()
            if d.is_dir() and not d.name.startswith("_") and d.name != "overview"
        )

        if patient_ids is not None:
            selected = [d for d in all_patients if d.name in patient_ids]
        else:
            selected = all_patients

        self.pairs = []
        for pdir in selected:
            ct_dir = pdir / "ct"
            mr_dir = pdir / "mr"
            ct_imgs = list_images(ct_dir)
            mr_imgs = list_images(mr_dir)
            common = sorted(set(ct_imgs) & set(mr_imgs))
            for name in common:
                self.pairs.append((ct_imgs[name], mr_imgs[name], f"{pdir.name}/{name}"))

        if not self.pairs:
            raise ValueError(f"No paired CT-MRI slices found in {data_root}")

        self.image_size = image_size
        self.augment = augment
        print(f"Loaded {len(self.pairs)} paired slices from {len(selected)} patients")

    def __len__(self):
        return len(self.pairs)

    def _load_image(self, path):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Cannot read: {path}")
        if (img.shape[0], img.shape[1]) != (self.image_size, self.image_size):
            img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        return img.astype(np.float32) / 127.5 - 1.0  # normalize to [-1, 1]

    def _augment(self, ct, mr):
        if random.random() < 0.5:
            ct = np.fliplr(ct).copy()
            mr = np.fliplr(mr).copy()
        if random.random() < 0.3:
            ct = np.flipud(ct).copy()
            mr = np.flipud(mr).copy()
        if random.random() < 0.3:
            k = random.randint(1, 3)
            ct = np.rot90(ct, k).copy()
            mr = np.rot90(mr, k).copy()
        if random.random() < 0.2:
            ct = ct * (0.9 + random.random() * 0.2)
            mr = mr * (0.9 + random.random() * 0.2)
        return ct, mr

    def __getitem__(self, idx):
        ct_path, mr_path, name = self.pairs[idx]
        ct = self._load_image(ct_path)
        mr = self._load_image(mr_path)

        if self.augment:
            ct, mr = self._augment(ct, mr)

        ct = np.clip(ct, -1, 1)
        mr = np.clip(mr, -1, 1)

        return (
            torch.from_numpy(ct).unsqueeze(0).float(),
            torch.from_numpy(mr).unsqueeze(0).float(),
            name,
        )


def create_dataloaders(data_root, image_size=128, batch_size=16,
                       val_ratio=0.10, test_ratio=0.10, num_workers=0,
                       seed=42):
    """
    Create train/val/test dataloaders with patient-level split.

    Returns:
        train_loader, val_loader, test_loader,
        test_pids, val_pids, train_pids
    """
    data_root = Path(data_root)
    all_patients = sorted(
        d.name for d in data_root.iterdir()
        if d.is_dir() and not d.name.startswith("_") and d.name != "overview"
    )

    random.seed(seed)
    random.shuffle(all_patients)

    n_total = len(all_patients)
    n_test = max(1, int(n_total * test_ratio))
    n_val = max(1, int(n_total * val_ratio))
    n_train = n_total - n_val - n_test

    test_pids = set(all_patients[:n_test])
    val_pids = set(all_patients[n_test:n_test + n_val])
    train_pids = set(all_patients[n_test + n_val:])

    print(f"Split: Train={n_train} patients, Val={n_val}, Test={n_test}")

    train_ds = PairedCTMRIDataset(data_root, image_size=image_size, augment=True, patient_ids=train_pids)
    val_ds = PairedCTMRIDataset(data_root, image_size=image_size, augment=False, patient_ids=val_pids)
    test_ds = PairedCTMRIDataset(data_root, image_size=image_size, augment=False, patient_ids=test_pids)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, test_pids, val_pids, train_pids
