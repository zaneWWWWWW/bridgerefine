"""Download, verify and restore the complete private BridgeRefine materials."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import zipfile

REPO = 'zaneWWWWWW/bridgerefine'
TAG = 'materials-2026-09-17'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def safe_path(root, name):
    rel = PurePosixPath(name)
    if rel.is_absolute() or '..' in rel.parts or not rel.parts or rel.parts[0] != 'materials':
        raise ValueError(f'Unsafe archive path: {name}')
    path = root.joinpath(*rel.parts)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Path escapes destination: {name}')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--cache', type=Path, help='Directory to keep downloaded ZIP archives')
    parser.add_argument('--extract-only', action='store_true', help='Use an already downloaded complete cache')
    parser.add_argument('--repo', default=REPO)
    parser.add_argument('--tag', default=TAG)
    args = parser.parse_args()
    destination = args.destination.resolve()
    cache = (args.cache or destination / '.materials-cache').resolve()
    cache.mkdir(parents=True, exist_ok=True)
    if not args.extract_only:
        subprocess.run(['gh', 'release', 'download', args.tag, '--repo', args.repo,
                        '--dir', str(cache), '--pattern', 'materials_index.json', '--clobber'], check=True)
    index = json.loads((cache / 'materials_index.json').read_text())
    archives = index['archives']
    for item in archives:
        name = item['name']
        if Path(name).name != name:
            raise ValueError('Invalid asset name')
        path = cache / name
        valid = path.is_file() and path.stat().st_size == item['bytes'] and digest(path) == item['sha256']
        if not valid and not args.extract_only:
            subprocess.run(['gh', 'release', 'download', args.tag, '--repo', args.repo,
                            '--dir', str(cache), '--pattern', name, '--clobber'], check=True)
            valid = path.stat().st_size == item['bytes'] and digest(path) == item['sha256']
        if not valid:
            raise ValueError(f'Missing or invalid ZIP: {name}')
        print(f'Verified {name}', flush=True)

    # Preflight all collisions before writing any material file.
    for item in archives:
        with zipfile.ZipFile(cache / item['name']) as archive:
            for member in archive.infolist():
                target = safe_path(destination, member.filename)
                if member.is_dir():
                    continue
                if target.is_symlink():
                    raise FileExistsError(f'Regular file destination is a symlink: {target}')
                if target.exists():
                    if not target.is_file() or target.stat().st_size != member.file_size:
                        raise FileExistsError(f'Existing file differs: {target}')
                    with archive.open(member) as stream:
                        expected = hashlib.file_digest(stream, 'sha256').hexdigest()
                    if digest(target) != expected:
                        raise FileExistsError(f'Existing file differs: {target}')
    for link in index['symlinks']:
        target = safe_path(destination, link['path'])
        resolved = (target.parent / link['target']).resolve()
        if not resolved.is_relative_to(destination):
            raise ValueError('Unsafe symbolic-link target')
        if target.is_symlink():
            if os.readlink(target) != link['target']:
                raise FileExistsError(f'Existing link differs: {target}')
        elif target.exists():
            raise FileExistsError(f'Expected a symbolic link: {target}')

    count = 0
    for item in archives:
        with zipfile.ZipFile(cache / item['name']) as archive:
            for member in archive.infolist():
                target = safe_path(destination, member.filename)
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        with archive.open(member) as src, target.open('xb') as out:
                            shutil.copyfileobj(src, out, length=1024 * 1024)
                    except Exception:
                        target.unlink(missing_ok=True)
                        raise
                count += 1
        print(f'Restored {item["name"]}', flush=True)
    for link in index['symlinks']:
        target = safe_path(destination, link['path'])
        if not target.is_symlink():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(link['target'], target_is_directory=link['is_directory'])
    print(f'Complete: {count} files and {len(index["symlinks"])} links in {destination / "materials"}')


if __name__ == '__main__':
    main()
