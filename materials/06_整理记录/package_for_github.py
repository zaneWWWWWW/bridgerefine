"""Prepare browseable Git copies plus a complete set of independent Release ZIPs."""
from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / '02_代码/bridgerefine'
BROWSE = REPO / 'materials'
OUTPUT = ROOT.parent / 'BridgeRefine_GitHub上传包_20260917'
LIMIT = 900 * 1024 * 1024


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def keep_in_git(rel, path):
    if rel.parts[0] == '04_数据':
        return rel.parts[1] not in {'04_影像数据', '06_模型权重', '07_预测缓存'} and path.suffix.lower() in {
            '.md', '.csv', '.json', '.txt', '.log', '.yaml', '.yml', '.png'}
    return path.suffix.lower() not in {'.tiff', '.tif'}


def main():
    if OUTPUT.exists() or BROWSE.exists():
        raise FileExistsError('Package/output already exists; preserve it and use a new location')
    OUTPUT.mkdir()
    entries = []
    links = []
    totals = defaultdict(lambda: dict(files=0, bytes=0))
    # Prune code (which contains the new browsing copies) at the directory level.
    for base, dirs, names in os.walk(ROOT, followlinks=False):
        base = Path(base)
        dirs[:] = sorted(d for d in dirs if d not in {'.git', '__pycache__'}
                         and not (base == ROOT and d == '02_代码'))
        for name in sorted(list(dirs) + names):
            p = base / name
            if p.is_symlink():
                rel = p.relative_to(ROOT)
                links.append(dict(path='materials/' + rel.as_posix(), target=os.readlink(p), is_directory=p.is_dir()))
        for name in sorted(names):
            p = base / name
            if not p.is_file() or p.is_symlink():
                continue
            rel = p.relative_to(ROOT)
            entries.append((rel, p))
            totals[rel.parts[0]]['files'] += 1
            totals[rel.parts[0]]['bytes'] += p.stat().st_size
    entries.sort(key=lambda item: item[0].as_posix())
    print(f'Found {len(entries)} files, {sum(p.stat().st_size for _, p in entries):,} bytes, {len(links)} links', flush=True)

    git_files = 0
    for rel, p in entries:
        if keep_in_git(rel, p):
            target = BROWSE / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)
            git_files += 1
    for item in links:
        rel = Path(item['path']).relative_to('materials')
        if rel.parts[0] in {'01_手稿', '03_图表'}:
            target = BROWSE / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(item['target'], target_is_directory=item['is_directory'])
    # Change only the remote browsing guide's code links; code lives at repo root.
    guide = BROWSE / '00_使用说明.md'
    content = guide.read_text()
    content = content.replace('(02_代码/bridgerefine/README.md)', '(../README.md)')
    content = content.replace('(02_代码/bridgerefine/)', '(../)')
    guide.write_text(content)
    # The ZIP copy must match the browsing copy for conflict-free restoration.
    entries = [(rel, guide if str(rel) == '00_使用说明.md' else p) for rel, p in entries]

    parts = []
    batch = []
    size = 0
    for rel, p in entries:
        n = p.stat().st_size
        if batch and size + n > LIMIT:
            parts.append(batch)
            batch = []
            size = 0
        if n > LIMIT:
            raise ValueError(f'Single file too large for target ZIP: {rel}')
        batch.append((rel, p))
        size += n
    if batch:
        parts.append(batch)
    archives = []
    for number, batch in enumerate(parts, 1):
        name = f'bridgerefine-materials-{number:02d}.zip'
        path = OUTPUT / name
        t0 = time.monotonic()
        with zipfile.ZipFile(path, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
            for rel, p in batch:
                already_compressed = p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.gz', '.npz', '.pdf', '.docx', '.pptx', '.xlsx'}
                archive.write(p, 'materials/' + rel.as_posix(),
                              compress_type=zipfile.ZIP_STORED if already_compressed else zipfile.ZIP_DEFLATED,
                              compresslevel=1)
        item = dict(name=name, bytes=path.stat().st_size, sha256=sha256(path), files=len(batch),
                    uncompressed_bytes=sum(p.stat().st_size for _, p in batch),
                    first_path=str(batch[0][0]), last_path=str(batch[-1][0]))
        archives.append(item)
        print(f'Packaged {number}/{len(parts)}: {name}, {item["bytes"]:,} bytes, {len(batch)} files, {time.monotonic()-t0:.1f}s', flush=True)
    index = dict(repo='zaneWWWWWW/bridgerefine', tag='materials-2026-09-17',
                 total_files=len(entries), total_bytes=sum(p.stat().st_size for _, p in entries),
                 git_browsing_files=git_files, categories=dict(totals), archives=archives, symlinks=links,
                 note='Complete curated non-code materials. Code is in the repository root. ZIPs are independent; restore all into one directory.')
    index_path = OUTPUT / 'materials_index.json'
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + '\n')
    checksums = archives + [dict(name=index_path.name, sha256=sha256(index_path))]
    (OUTPUT / 'SHA256SUMS.txt').write_text(''.join(f'{x["sha256"]}  {x["name"]}\n' for x in checksums))
    # Keep an index in Git as a second copy of the Release checksums.
    shutil.copy2(index_path, REPO / 'materials_index.json')
    print(f'Complete: {len(archives)} ZIP files; upload size {sum(x["bytes"] for x in archives):,} bytes', flush=True)


if __name__ == '__main__':
    main()
