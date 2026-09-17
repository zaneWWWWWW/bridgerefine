# BridgeRefine JIIM Submission Audit

## Superseding Audit: 2026-09-08

The archived benchmark script uses 20 sampling steps and five recursions, repeats one preloaded CT slice 100 times per repetition, and excludes input transfer and model loading. Its 420-fold ratio does not characterize the final 10x2 reconstruction configuration. Peak allocation includes resident tensors and models; the CT-only model remains resident for bridge measurement. Values 44.6/144.7 MB must not be called additional activation memory. A matched 10x2 timing and isolated-memory benchmark remains required. The old joint accuracy/latency figures have been removed from the manuscript rather than relabeled as a matched experiment.

Patient arrays independently confirm improvement over MG-CycleGAN in 17/18 patients, and over SelfRDB, SynDiff, and BridgeGAN in 18/18. CT-only comparison has two-sided Wilcoxon p=0.0001068115234375 (one-sided p=0.00005340576171875). This supersedes earlier ambiguous direction labels.

The coarse image is derived from CT and does not supply independently acquired patient information. Interpret the gain as a benefit of a learned intermediate representation, not recovery of information absent from CT. Earlier statements below concerning a matched 420-fold trade-off are superseded by this audit.

## One-Sentence Paper Story

BridgeRefine tests whether a frozen diffusion bridge adds measurable information beyond direct CT-to-synthetic-MRI regression and shows a modest patient-level benefit at approximately 420-fold inference latency.

## Section Outline

1. **Background:** define paired CT-to-synthetic-MRI reconstruction and the unresolved incremental-value question.
2. **Materials and Methods:** describe the patient-level split, unified coarse protocol, three input configurations, metrics, statistics, and efficiency benchmark.
3. **Results:** report fixed checkpoints, input ablation, matched patient effects, three seeds, negative loss/adversarial findings, efficiency, and qualitative examples.
4. **Discussion:** interpret component responsibility, metric-dependent rankings, supervision findings, practical cost, clinical boundaries, and limitations.
5. **Conclusion:** state the modest bridge contribution and the accuracy--efficiency trade-off without diagnostic claims.

## Claim--Evidence Map

| Claim | Evidence | Status |
|---|---|---|
| Supervised refinement is the principal observed performance source | SelfRDB 0.5855 versus BridgeRefine-L1 0.8184 | Supported |
| CT is the dominant information source | CT-only 0.8095 versus coarse-only 0.7823 | Supported |
| Coarse MRI adds information beyond CT | CT+coarse 0.8184 versus CT-only 0.8095; paired delta 0.0089 | Supported at seed 42 |
| Patient-level improvement is directionally consistent | 17/18 improved; bootstrap CI 0.0058--0.0122; raw Wilcoxon p=1.07e-4 | Supported |
| Ordering persists across runs | BR-L1 range 0.8147--0.8184 versus CT-only 0.7883--0.8095 | Supported descriptively |
| Gradient loss is a core contribution | Shared-seed direction changes | Not supported; excluded |
| Tested adversarial refiners improve SSIM | BridgeGAN 0.8046 versus BR-L1 0.8184 | Not supported; negative result |
| BridgeRefine is low latency | 1017 ms versus 2.42 ms | Refuted; trade-off reported |
| Synthetic MRI is diagnostically equivalent to acquired MRI | No reader, lesion, downstream, or external evidence | Not supported; explicitly excluded |

## Benchmark-Paper Alignment

The manuscript follows Lai et al. in five respects:

1. One experimental question organizes the paper.
2. Evaluation uses complementary metrics rather than one headline score.
3. Negative findings are reported as results, not hidden.
4. Computational cost and evaluation artifacts are described.
5. Discussion distinguishes image quality from clinical utility.

The manuscript does not copy Lai et al.'s scientific claims or structure verbatim. Their question concerns training-set size in unconditional MRI synthesis; BridgeRefine concerns component value in paired CT-to-MRI translation.

## Statistical Consistency Note

The frozen handoff summary reported CT-only run SD as 0.0093. Applying a consistent sample-standard-deviation calculation to the recorded run means 0.8095, 0.8050, and 0.7883 yields 0.0112. The JIIM draft reports 0.0112 and explicitly identifies it as sample SD across three runs. BridgeRefine-L1 sample SD is 0.0021. Raw unrounded CT-only values should be recovered if available before final submission.

The primary matched CT-only result file contains two-sided raw Wilcoxon p=1.068e-4 and bootstrap CI [0.0058, 0.0122]. The main draft reports the raw p-value and does not claim a CT-only Holm-adjusted p-value. Holm adjustment is retained only for the separate four-baseline family. The archived handoff manifest's CT-only adjusted p=0.0002 is not used until its comparison family is documented.

## Configuration-Specific Efficiency Caveat

The main reconstruction table uses 10 bridge sampling steps and two recursive estimates. The available timing benchmark uses 20 steps and five recursive estimates and repeats one preloaded CT slice. It provides a recorded high-cost deployment reference, not a matched latency for the final reconstruction configuration. The manuscript explicitly states this limitation and does not use the 420-fold ratio as a final-model efficiency claim.

## Five-Dimension Self-Review

### Contribution

- Pass: contribution is a testable workflow decomposition with patient-level evidence.
- Risk: method novelty is limited; title and cover letter must emphasize validation insight.

### Writing Clarity

- Pass: Background follows task -> challenge -> gap -> method -> study questions.
- Pass: every Methods subsection defines purpose, implementation, and interpretation.

### Experimental Strength

- Pass: strong direct baseline, four generative comparators, input ablation, seeds, patient statistics, and efficiency.
- Risk: effect over CT-only is modest and must remain numerically explicit.

### Evaluation Completeness

- Pass: multiple paired reconstruction dimensions and failure configurations are included.
- Risk: no external cohort, reader study, lesion task, or full multi-seed non-SSIM evaluation.

### Method Design Soundness

- Pass: frozen bridge and unified 10x2 protocol isolate intermediate input.
- Risk: 420x latency may outweigh benefit for deployment; CT-only remains a serious alternative.

## Submission Blockers

- [ ] Author names, affiliations, ORCID identifiers, and correspondence.
- [ ] Ethics approval/exemption and consent language.
- [ ] De-identification and governance statement.
- [ ] Scanner, acquisition, sequence, slice-thickness, and registration metadata.
- [ ] CT-to-MRI acquisition interval and inclusion/exclusion criteria.
- [ ] Funding, competing interests, author contributions, and acknowledgements.
- [ ] Data and code availability text approved by the data owner.
- [ ] Recover exact CT-only run precision or approve 0.0112 from recorded means.
- [ ] Confirm the statistical family behind the archived adjusted p=0.0002.
- [ ] Verify all references against DOI metadata at final submission.
