"""Curate existing BridgeRefine artifacts without modifying the originals."""
from pathlib import Path
import csv
import hashlib
import json
import os
import shutil
import zipfile

BASE = Path(__file__).resolve().parents[2]
DEST = Path(__file__).resolve().parents[1]
PROJECT = BASE / 'Cross-modal conversion'
V2 = BASE / 'BridgeRefine_paper_handoff_v2/handoff_v2'
FROZEN = BASE / 'BridgeRefine_paper_handoff_v2/handoff_send/final_frozen_light'
JIIM = PROJECT / 'other_model/paper/jiim_submission'
REPO = DEST / '02_代码/bridgerefine'
records = []


def record(target, source, digest):
    records.append(dict(path=str(target.relative_to(DEST)), source=source,
                        bytes=target.stat().st_size, sha256=digest))


def copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    shutil.copy2(source, target)
    with target.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    record(target, str(source.relative_to(BASE)), digest)


def tree(source, target, predicate=lambda p: True):
    for p in sorted(source.rglob('*')):
        if p.is_file() and not p.is_symlink() and predicate(p):
            copy(p, target / p.relative_to(source))


def link(target, source):
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(os.path.relpath(source, target.parent), target_is_directory=source.is_dir())


def unzip_file(archive, member, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    digest = hashlib.sha256()
    with archive.open(member) as src, target.open('wb') as out:
        while chunk := src.read(1024 * 1024):
            out.write(chunk)
            digest.update(chunk)
    record(target, 'Cross-modal conversion.zip::' + member, digest.hexdigest())
    return digest.hexdigest()


def main():
    print('1/7 Collecting manuscripts, figures and original result tables', flush=True)
    for ext in ['tex', 'pdf']:
        copy(JIIM / f'BridgeRefine_JIIM_中文稿.{ext}', DEST / f'01_手稿/中文主稿/BridgeRefine_JIIM_中文稿.{ext}')
    for stem in ['full', 'blinded', 'body', 'supplement']:
        for ext in ['tex', 'pdf']:
            p = JIIM / f'BridgeRefine_JIIM_{stem}.{ext}'
            if p.exists():
                copy(p, DEST / '01_手稿/英文投稿旧稿_待同步' / p.name)
    for name in ['Paper_English_Draft.md', '论文_中文稿.md']:
        copy(BASE / name, DEST / '01_手稿/中英文工作稿_20260911' / name)
    tree(V2 / 'paper', DEST / '01_手稿/审计交接稿_20260910')
    copy(PROJECT / 'output/BridgeRefine_Chinese_manuscript.docx', DEST / '01_手稿/历史Word稿_仅供参考/BridgeRefine_Chinese_manuscript.docx')
    tree(JIIM / 'figures', DEST / '03_图表/投稿配套图', lambda p: p.suffix != '.py')
    tree(FROZEN / 'figures', DEST / '03_图表/交接版配套图')
    tree(FROZEN / 'tables', DEST / '03_图表/历史表格_按来源保留')
    tree(PROJECT / 'other_model/paper/figma_pipeline_design', DEST / '03_图表/流程图设计材料')
    tree(PROJECT / 'output/figures', DEST / '03_图表/机制示意图', lambda p: p.suffix != '.py')
    for sub in ['中文主稿', '英文投稿旧稿_待同步']:
        link(DEST / '01_手稿' / sub / 'figures', DEST / '03_图表/投稿配套图')
    link(DEST / '01_手稿/审计交接稿_20260910/figures', DEST / '03_图表/交接版配套图')
    tree(V2 / 'metrics', DEST / '04_数据/01_最终指标/metrics')
    tree(V2 / 'efficiency', DEST / '04_数据/01_最终指标/efficiency', lambda p: p.suffix != '.py')
    tree(V2 / 'qualitative_cases', DEST / '04_数据/05_定性病例')
    tree(V2 / 'figures_source', DEST / '03_图表/审计交接原始图')
    link(DEST / '03_图表/真实定性病例', DEST / '04_数据/05_定性病例')
    for source, name in [(FROZEN / 'source_data', '交接冻结版'), (JIIM / 'source_data', '投稿版')]:
        tree(source, DEST / '04_数据/02_原始结果表' / name,
             lambda p: p.suffix.lower() in {'.csv', '.json', '.txt', '.md'})
    print('2/7 Collecting experiment records and prediction caches', flush=True)
    tree(BASE / 'BridgeRefine_paper_handoff_v2/supplement', DEST / '04_数据/03_逐实验记录')
    for source, name in [(JIIM / 'source_data/unified10x2/test_coarse_10x2.pt', 'test_coarse_10x2.pt'),
                         (PROJECT / 'other_model/paper/improved/predictions.pt', 'legacy_predictions.pt')]:
        if source.exists():
            copy(source, DEST / '04_数据/07_预测缓存' / name)
    print('3/7 Copying paired image slices and patient split', flush=True)
    tree(PROJECT / 'data_slices', DEST / '04_数据/04_影像数据/data_slices')
    copy(PROJECT / 'checkpoints/selfrdb/patient_split.json', DEST / '04_数据/04_影像数据/patient_split.json')
    print('4/7 Recovering raw volumes and matching checkpoint hashes from ZIP', flush=True)
    checkpoint_rows = list(csv.DictReader((V2 / 'checkpoints/checkpoints_manifest.csv').open(encoding='utf-8-sig')))
    restored = []
    with zipfile.ZipFile(BASE / 'Cross-modal conversion.zip') as archive:
        members = set(archive.namelist())
        for member in sorted(members):
            prefix = 'Cross-modal conversion/brain/'
            if member.startswith(prefix) and not member.endswith('/'):
                relative = Path(member.removeprefix(prefix))
                if relative.is_absolute() or '..' in relative.parts:
                    raise ValueError(member)
                unzip_file(archive, member, DEST / '04_数据/04_影像数据/brain' / relative)
        supplements = {'42': 'input_ct_coarse_seed42', '123': 'input_ct_coarse_seed123', '2026': 'input_ct_coarse_seed2026'}
        for row in checkpoint_rows:
            original = row['checkpoint_path'].replace('\\', '/').split('Cross-modal conversion/', 1)[1]
            target = DEST / '04_数据/06_模型权重' / original
            member = 'Cross-modal conversion/' + original
            if row['method'] == 'BridgeRefine-L1':
                source = BASE / 'BridgeRefine_paper_handoff_v2/supplement' / supplements[row['seed']] / 'checkpoints/best.pt'
                copy(source, target)
                actual = records[-1]['sha256']
            elif member in members:
                actual = unzip_file(archive, member, target)
            else:
                raise FileNotFoundError(member)
            if actual != row['sha256']:
                raise ValueError(f'Checkpoint hash mismatch: {original}')
            restored.append({**row, 'local_path': str(target.relative_to(DEST)), 'verified_sha256': actual, 'status': 'match'})
    coarse = DEST / '04_数据/03_逐实验记录/input_coarse_only_seed42/checkpoints/best.pt'
    link(DEST / '04_数据/06_模型权重/other_model/input_ablation/coarse_only_seed42/best.pt', coarse)
    output = DEST / '06_整理记录/权重核对.csv'
    with output.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(restored[0])); writer.writeheader(); writer.writerows(restored)
    print('5/7 Collecting code, configs and provenance notes', flush=True)
    root_code = ['main.py', 'preprocess_all.py', 'diffusion_dataset.py', 'diffusion_model.py',
                 'diffusion_train.py', 'diffusion_inference.py', 'evaluate_metrics.py']
    for name in root_code:
        p = PROJECT / name
        if p.exists(): copy(p, REPO / 'original/project' / name)
    for name in ['selfrdb', 'mg_cyclegan', 'syndiff']:
        tree(PROJECT / name, REPO / 'original/project' / name, lambda p: p.suffix == '.py')
    for name in ['SelfRDB', 'BridgeGAN', 'MG_CycleGAN', 'SynDiff', 'Masked_Bridge', 'Masked_Adv_Diff_Bridge']:
        tree(PROJECT / 'other_model' / name, REPO / 'original/project/other_model' / name, lambda p: p.suffix == '.py')
    for p in sorted((PROJECT / 'other_model').glob('*.py')):
        if p.name.startswith('make_'): continue
        copy(p, REPO / 'original/project/other_model' / p.name)
    tree(PROJECT / 'experiments', REPO / 'original/project/experiments', lambda p: p.suffix == '.py')
    tree(V2 / 'scripts', REPO / 'original/handoff_v2/scripts')
    copy(V2 / 'efficiency/benchmark.py', REPO / 'original/handoff_v2/efficiency/benchmark.py')
    tree(V2 / 'configs', REPO / 'configs')
    for source, name in [(PROJECT / 'other_model/paper/nature_bridge_refine/build_figures.py', 'build_figures_legacy.py'),
                         (JIIM / 'figures/redraw_pipeline.py', 'redraw_pipeline.py'),
                         (PROJECT / 'output/figures/make_bridgerefine_schematic.py', 'make_bridgerefine_schematic.py')]:
        copy(source, REPO / 'original/figure_scripts' / name)
    tree(V2 / 'metadata', DEST / '05_说明与历史/元数据缺项')
    for p in [V2 / 'README.md', V2 / 'MISSING.md', V2 / 'HANDOFF_EXECUTION.md', V2 / 'manifest.csv']:
        copy(p, DEST / '05_说明与历史/审计交接说明' / p.name)
    tree(V2 / 'checkpoints', DEST / '05_说明与历史/原始权重清单')
    for p in sorted(JIIM.glob('*.md')):
        copy(p, DEST / '05_说明与历史/投稿说明与审查' / p.name)
    for p in sorted(JIIM.glob('*_v*.pdf')):
        copy(p, DEST / '05_说明与历史/中文历史版本' / p.name)
    for name in ['交接补充说明.md', 'BridgeRefine_实验总览.md']:
        copy(BASE / name, DEST / '05_说明与历史' / name)
    tree(BASE / 'journal_benchmark', DEST / '05_说明与历史/期刊参考材料')
    print('6/7 Writing full file provenance and checksums', flush=True)
    with (DEST / '06_整理记录/文件来源清单.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['path', 'source', 'bytes', 'sha256'])
        writer.writeheader(); writer.writerows(records)
    with (DEST / '06_整理记录/原件副本_SHA256SUMS.txt').open('w') as f:
        for row in records:
            f.write(row['sha256'] + '  ' + row['path'] + '\n')
    code = [dict(path=r['path'].split('02_代码/bridgerefine/', 1)[1], source=r['source'], sha256=r['sha256'])
            for r in records if r['path'].startswith('02_代码/bridgerefine/')]
    (REPO / 'SOURCE_MANIFEST.json').write_text(json.dumps(code, ensure_ascii=False, indent=2) + '\n')
    summary = {'copied_files': len(records), 'copied_bytes': sum(r['bytes'] for r in records),
               'verified_primary_checkpoints': len(restored), 'originals_modified': False}
    (DEST / '06_整理记录/整理统计.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('7/7 Complete: ' + json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
