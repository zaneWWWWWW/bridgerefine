# -*- coding: utf-8 -*-
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
    return {path.stem: path for path in sorted(folder.iterdir()) if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS}

def extract_canny_edge(img, low=50, high=150):
    img_u8 = ((img + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    edges = cv2.Canny(img_u8, low, high)
    return edges.astype(np.float32) / 255.0

class MultiPatientDataset(Dataset):
    def __init__(self, data_root, image_size=128, augment=False, patient_dirs=None):
        data_root = Path(data_root)
        all_patients = sorted(d for d in data_root.iterdir() if d.is_dir() and not d.name.startswith("_"))
        if patient_dirs is not None:
            selected = [d for d in all_patients if d.name in patient_dirs]
        else:
            selected = all_patients
        self.pairs = []
        for pdir in selected:
            ct_imgs = list_images(pdir / "ct")
            mr_imgs = list_images(pdir / "mr")
            common = sorted(set(ct_imgs) & set(mr_imgs))
            for name in common:
                self.pairs.append((ct_imgs[name], mr_imgs[name], f"{pdir.name}/{name}"))
        if not self.pairs:
            raise ValueError(f"No paired images found")
        self.image_size = image_size
        self.augment = augment
        print(f"Loaded {len(self.pairs)} slices from {len(selected)} patients")

    def __len__(self):
        return len(self.pairs)

    def _load_image(self, path):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Cannot read: {path}")
        img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        return img.astype(np.float32) / 127.5 - 1.0

    def _augment(self, ct, mr, edge):
        if random.random() < 0.5:
            ct, mr, edge = np.fliplr(ct).copy(), np.fliplr(mr).copy(), np.fliplr(edge).copy()
        if random.random() < 0.3:
            ct, mr, edge = np.flipud(ct).copy(), np.flipud(mr).copy(), np.flipud(edge).copy()
        if random.random() < 0.3:
            ct = ct * (0.9 + random.random() * 0.2)
        return ct, mr, edge

    def __getitem__(self, idx):
        ct_path, mr_path, name = self.pairs[idx]
        ct = self._load_image(ct_path)
        mr = self._load_image(mr_path)
        edge = extract_canny_edge(ct)
        if self.augment:
            ct, mr, edge = self._augment(ct, mr, edge)
        ct, mr = np.clip(ct, -1, 1), np.clip(mr, -1, 1)
        return (
            torch.from_numpy(ct).unsqueeze(0).float(),
            torch.from_numpy(mr).unsqueeze(0).float(),
            torch.from_numpy(edge).unsqueeze(0).float(),
            name,
        )

def create_dataloaders(data_root, image_size=128, batch_size=16, val_ratio=0.1, test_ratio=0.1, num_workers=0):
    data_root = Path(data_root)
    all_patients = sorted(d.name for d in data_root.iterdir() if d.is_dir() and not d.name.startswith("_"))
    random.seed(42)
    random.shuffle(all_patients)
    n_total = len(all_patients)
    n_test = max(1, int(n_total * test_ratio))
    n_val = max(1, int(n_total * val_ratio))
    n_train = n_total - n_val - n_test
    test_patients = set(all_patients[:n_test])
    val_patients = set(all_patients[n_test:n_test + n_val])
    train_patients = set(all_patients[n_test + n_val:])
    print(f"Split by patient: Train={n_train}, Val={n_val}, Test={n_test}")
    train_ds = MultiPatientDataset(data_root, image_size=image_size, augment=True, patient_dirs=train_patients)
    val_ds = MultiPatientDataset(data_root, image_size=image_size, augment=False, patient_dirs=val_patients)
    test_ds = MultiPatientDataset(data_root, image_size=image_size, augment=False, patient_dirs=test_patients)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader, test_patients, val_patients, train_patients
