# JIIM 投稿准备清单

## 已完成

- [x] JIIM 风格英文完整稿
- [x] Double-blind 匿名稿
- [x] 244-word structured-style abstract
- [x] 6 个关键词
- [x] Patient-level fixed-checkpoint comparison
- [x] CT-only/coarse-only/CT+coarse ablation
- [x] Three-seed comparison
- [x] Patient-level paired Wilcoxon and bootstrap CI
- [x] Efficiency benchmark with synchronized CUDA timing
- [x] Negative Gradient and adversarial findings
- [x] Supplementary material for exploratory subset/pilot experiments
- [x] Final source data copied from frozen handoff package
- [x] DOI metadata for newly added citations checked
- [x] Full and blinded PDFs compiled successfully
- [x] Full and blinded PDFs visually inspected

## Must complete before submission

- [ ] Replace the red author placeholder with author names, affiliations, ORCID, and corresponding-author email.
- [ ] Replace ethics placeholder with committee, approval/exemption number, and consent language.
- [ ] Add de-identification and data-governance statement.
- [ ] Add CT/MRI scanner, sequence, acquisition, and slice-thickness metadata.
- [ ] Add CT-to-MRI acquisition interval and patient inclusion/exclusion criteria.
- [ ] Add registration software, transformation type, and quality-control procedure.
- [ ] Add funding, competing interests, acknowledgements, and CRediT contributions.
- [ ] Confirm data-availability wording with the data owner.
- [ ] Confirm code-availability wording and add the final source archive SHA-256 or public repository commit.
- [ ] Confirm the statistical family behind the archived Holm-adjusted p=0.0002.
- [ ] Recover exact unrounded CT-only seed means or formally approve the sample SD reported in the draft.
- [ ] Recheck all DOI metadata and references at submission time.
- [ ] Export a final single-column Word or LaTeX version matching the current JIIM instructions.

## Files to submit

### Full version

`BridgeRefine_JIIM_full.tex` and `BridgeRefine_JIIM_full.pdf`

### Blinded version

`BridgeRefine_JIIM_blinded.tex` and `BridgeRefine_JIIM_blinded.pdf`

### Supplementary material

`BridgeRefine_JIIM_supplement.tex` and `BridgeRefine_JIIM_supplement.pdf`

### Supporting data

- `source_data/`
- `figures/`
- experiment handoff package: `/home/zanewang/projects/paper/BridgeRefine_交接包/`

## Journal-specific points

JIIM is double blind, so the anonymous manuscript must omit author names, affiliations, self-identifying acknowledgements, and identifying institution names in the running text. The full manuscript should contain the title page and all declarations. The journal's current instructions request a 150--250 word abstract and 4--6 keywords; the draft satisfies both limits.

