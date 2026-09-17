# BridgeRefine JIIM Submission Draft

## Files

- `BridgeRefine_JIIM_full.tex` / `.pdf`: full manuscript draft with visible submission blockers.
- `BridgeRefine_JIIM_blinded.tex` / `.pdf`: double-blind review draft without author information.
- `BridgeRefine_JIIM_body.tex`: shared manuscript body.
- `BridgeRefine_JIIM_supplement.tex` / `.pdf`: exploratory loss and adversarial pilot material.
- `BridgeRefine_JIIM_中文稿.tex` / `.pdf`: Chinese reading draft aligned with the English submission structure.
- `中文阅读与修改建议.md`: Chinese review notes and items requiring verification.
- `JIIM_SUBMISSION_AUDIT.md`: claim--evidence map, benchmark alignment, statistical notes, and blockers.
- `JIIM_READY_CHECKLIST.md`: final submission checklist and JIIM-specific file requirements.
- `source_data/`: frozen machine-readable result artifacts.
- `figures/`: copied frozen publication figures.

## Journal Alignment

The draft follows the official JIIM data-driven research order:

1. Abstract
2. Background
3. Materials and Methods
4. Results
5. Discussion
6. Conclusion
7. Acknowledgements
8. Statements and Declarations
9. References

The abstract contains 244 words and six keywords. The manuscript is single-column, uses no more than three heading levels, and has both full and blinded wrappers. JIIM accepts LaTeX for manuscripts with mathematical content.

## Benchmark Alignment

The writing strategy follows Lai et al., *Generating Brain MRI with StyleGAN2-ADA: The Effect of the Training Set Size on the Quality of Synthetic Images*:

- one experimental question organizes the paper;
- complementary evaluation dimensions are reported;
- negative results and metric limitations are retained;
- computational cost is treated as a scientific result;
- clinical utility is separated from image quality.

BridgeRefine's question is different: it tests whether a frozen diffusion bridge adds information beyond direct CT regression and whether the gain justifies latency.

## Compile

```bash
xelatex BridgeRefine_JIIM_full.tex
xelatex BridgeRefine_JIIM_full.tex

xelatex BridgeRefine_JIIM_blinded.tex
xelatex BridgeRefine_JIIM_blinded.tex

xelatex BridgeRefine_JIIM_supplement.tex
xelatex BridgeRefine_JIIM_supplement.tex
```

## Important Statistical Change

The frozen handoff reported CT-only run SD as `0.0093`. A consistent sample-SD calculation from the recorded run means `0.8095, 0.8050, 0.7883` gives `0.0112`. The JIIM draft reports `0.0112` and explicitly labels it sample SD across three runs. Recover unrounded run means before submission if possible.

The primary CT-only result is reported using the raw paired Wilcoxon value (`p=1.07e-4`) and bootstrap CI. Holm adjustment is reported for the separate four-baseline family. Confirm the archived CT-only adjusted p-value's comparison family before using `p=0.0002` in the submission.

## Not Submission-Ready Until

- all red `[REQUIRED BEFORE SUBMISSION: ...]` fields are replaced with verified information;
- ethics, consent, de-identification, acquisition, and registration details are complete;
- funding, competing interests, author contributions, and data/code availability are approved;
- references receive a final DOI audit;
- a target-year JCR/institutional journal check is complete.
