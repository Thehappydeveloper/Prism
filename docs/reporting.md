# Reports

Prism reports are versioned analysis builds over immutable run artifacts. A
definition lives in `docs/argus_mmsys2027/reports/*.prism-report.json`; a build freezes
the selected run IDs and artifact fingerprints before producing outputs under
`docs/argus_mmsys2027/generated/<report-id>/<build-id>/`.

Use the **Reports** page for validation and builds, or use the CLI:

```bash
PYTHONPATH=src python -m reporting list
PYTHONPATH=src python -m reporting validate argus_mmsys2027 --results-root results
PYTHONPATH=src python -m reporting build argus_mmsys2027 --results-root results
```

For the recommended Argus reviewer workflow, use the wrapper instead. It
checks the runtime tools and the complete 2,219-run bundled cohort before
building the parent report, without requiring Node.js or the browser UI:

```bash
python src/scripts/reproduce_argus_mmsys2027.py --precomputed
```

Each build contains `inputs.json`, `runs.csv`, `tables.md`, and
`report-manifest.json`. The latter captures the definition, validation result,
and output inventory. Current report builds freeze and organize the remaining
paper inputs; each definition declares the table and figure renderer it needs.
