# Paper reproduction data

The paper batch plans require the dedicated MMSys paper data package, rather
than a generic FoV-trace download. It contains the exact s7 and s15 inputs:

- `s7.mpd` and `s15.mpd`;
- 2,700 PLY frames for each scene (90 seconds at 30 fps); and
- 27 verified MazeLab FoV traces: 14 for s7 and 13 for s15. Each CSV includes
  the viewer pose and per-tile ground-truth FoV scores used by the oracle.

The simulator reads the MPDs and PLY frames but does not decode V3C segment
units. Accordingly, the archive intentionally does not contain V3C media
segments: they are not needed to reproduce the paper runs.

The download is 37.2 GB in total (two 18.6 GB PLY archives plus small MPD and
trace files). Extraction produces about 228 GiB of PLY data. The fetcher
stages data beside `data/` to avoid a second extracted copy, but at least
270 GiB of free space is recommended.

Install the data with:

```bash
python src/scripts/get_argus_mmsys2027_data.py
```

The fetcher verifies the SHA-256 checksum of every downloaded file, installs
the data under `data/`, and registers the `s7` and `s15` profiles in the local
`.prism_settings/` directory. It also enables per-stage prediction timing,
which the process-scaling report requires. It leaves worker counts local to
the machine; all paper plans explicitly set their semantic segment-worker
configuration. It is idempotent: a complete installation is left untouched.
The fetcher pins the paper's published Zenodo version:
[Zenodo record 24118210](https://doi.org/10.5281/zenodo.24118210). Pass
`--record-id` only when intentionally installing another record version.

The dataset profile also records the MPD-wide tile extent summary (2,232 tile
bounds). This is useful for inspecting the dataset in the UI. It is not the
source of simulation geometry: Prism reads the per-segment tile bounds from
the downloaded MPD at run time.

The expected upload filenames and their SHA-256 checksums are:

| File | SHA-256 |
| --- | --- |
| `s7.zip` | `881490a27446c2166f827755ffe9ac015fef2d878799507a4a057106e219c853` |
| `S15.zip` | `23a7320b084a3e7e43e422749eada3c33fc2d5951c103d438caed7e467ed2c9b` |
| `s7.mpd` | `a94be2f201e448324f336cba3145c4e31e5cf2f3371a4204f60fb33deb52dc46` |
| `s15.mpd` | `8592b362843329359ccd8549b87a7418f98b085e56efd284418df5324223f161` |
| `mazelab_fov_traces.zip` | `ef854cfceff502893c89024de1fad69939aef2aacb7a5240ae8310165360e201` |
