# Paper asset scripts

These scripts were found in the author's working tree but were not included in
the original source snapshot. They generate manuscript figures or format the
Chinese Word manuscript. Their sources are recorded in
`PAPER_ASSET_MANIFEST.json` at the repository root.

- `jiim_submission/redraw_pipeline.py`: legacy pipeline figure for the JIIM
  submission.
- `nature_bridge_refine/build_figures.py`: figure builder for the Nature-style
  manuscript draft; it expects adjacent `phase1_results/` input data.
- `nature_bridge_refine/format_word.py`: formatting pass for the Chinese Word
  manuscript; input and output paths are Windows-specific and must be adjusted
  before use on another system.
- `output/figures/make_bridgerefine_schematic.py`: editable method schematic
  generator.

These are archival copies. Their original paths and contents are preserved;
they have not been refactored or tested as portable commands.
