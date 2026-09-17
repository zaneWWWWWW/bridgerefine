"""Verify collected original copies using the local provenance manifest."""
import argparse
import csv
import hashlib
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--quick', action='store_true', help='Check existence and size only')
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
failures = []
count = 0
with (root / '06_整理记录/文件来源清单.csv').open(encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        count += 1
        path = root / row['path']
        code_prefix = '02_代码/bridgerefine/'
        if not path.exists() and row['path'].startswith(code_prefix):
            path = root.parent / row['path'].removeprefix(code_prefix)
        if not path.is_file() or path.stat().st_size != int(row['bytes']):
            failures.append(row['path'])
            continue
        if not args.quick:
            with path.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            if digest != row['sha256']:
                failures.append(row['path'])
print(f'Checked {count} original-copy files; failures: {len(failures)}; SHA-256: {not args.quick}')
for item in failures:
    print(item)
raise SystemExit(bool(failures))
