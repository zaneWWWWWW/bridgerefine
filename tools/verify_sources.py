"""Check that archived source code/config files match their recorded SHA-256."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
records = json.loads((root / 'SOURCE_MANIFEST.json').read_text())
for row in records:
    path = root / row['path']
    with path.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if actual != row['sha256']:
        raise SystemExit(f'MISMATCH: {row["path"]}')
print(f'Verified {len(records)} original source/config files.')
