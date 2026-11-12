# Run artifacts and manifests

New completed runs write `manifest.json` using schema `prism.run-manifest.v1` before
their `completion.json` marker.  It records the concrete configuration, source revision,
input fingerprints, MPD-plus-PLY timeline, and generated artifacts.  Existing run folders
remain valid legacy artifacts and do not need a migration.

`results/_run_index.sqlite` is a rebuildable local catalogue of compact run metadata used by
the run and report interfaces. It is an acceleration layer, not the source of truth: deleting
it is safe because it can be rebuilt from the run directories.

To rebuild it explicitly after moving or restoring result folders:

```bash
PYTHONPATH=src python src/scripts/rebuild_run_index.py --results results
```
