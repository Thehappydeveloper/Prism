# Prism: V3C Component-Tiled Streaming Simulator (MMSys 2026)

Prism is an offline, trace-driven simulator for adaptive streaming of MPEG-I V3C / V-PCC content described by DASH-aligned MPDs.
It was developed as a research artifact for the MMSys 2026 Prism paper, and is intended for researchers and engineers who want to evaluate ABR logic, FoV prediction behavior, and QoE tradeoffs before deploying changes in production players.
Prism models viewport-aware and component-aware adaptation over spatially tiled V3C content, replays network and FoV traces, and writes run artifacts to `results/`.
It simulates the DASH control loop without decoding or rendering, so experiments stay reproducible and fast to iterate.

> **Reproducing the Argus MMSys 2027 paper?** Use the dedicated
> [Argus reproduction guide](docs/argus_mmsys2027/README.md), not the generic
> Quick Start below. It provides the exact paper plans and a one-command
> bundled-results workflow.

## Quick Start

### Requirements

- Python 3.11 or 3.12
- Node.js 18 or newer (only to build the web UI; Node 22 is the development version in `.nvmrc`)

Prism supports standard Python virtual environments on Linux, macOS, Windows, and WSL. No Nix, container, or automatic download is required.

### Create the environment

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.lock
cd src/web
npm ci
npm run build
cd ../..
python src/scripts/check_environment.py --ui
```

### Run the server
```bash
python src/server.py
```

Open `http://localhost:8000/`.

For model training, additionally install `requirements-training.lock` and run `python src/scripts/check_environment.py --training`. Add `--ui` only when validating the browser frontend too.

### Recommended preprocessing: Dataset Transformation Tool

Open `http://localhost:8000/static/dataset-transformation.html` before registering or running datasets that need coordinate cleanup.

The Dataset Transformation Tool can apply one shared transform to FoV traces, PLY point-cloud frames, or both:
- Select a reference PLY file, or provide a PLY sequence folder. For a folder, Prism uses the first numerically indexed PLY frame using the same frame-number extraction as dataset registration.
- Compute a height-based scale from the selected reference axis.
- Optionally swap the output height axis, for example `Z <-> Y`.
- Optionally shift coordinates into positive space.
- Optionally export MPEG PCC/TMC2-style integer voxel coordinates using geometry bit depth such as `vox9`, `vox10`, `vox11`, or `vox12`.
- Export normalized geometry, voxelized geometry, or both. Point-cloud sequence export supports a configurable worker count.

When registering a dataset exported by this tool, enable `Already normalized` in the dataset form. This stores an identity transform and prevents Prism from applying normalization a second time.

The same pipeline is available from the command line:
```bash
python src/scripts/dataset_transform.py \
  --trace-root data/fov_traces/example \
  --ply-input data/longdress/ply \
  --output data/longdress_transformed \
  --x-column HMDPX --y-column HMDPY --z-column HMDPZ \
  --height-axis y --output-height-axis y \
  --real-height-m 1.8 \
  --traces --point-clouds --positive \
  --export-mode both --voxel-bits 10 --workers 8
```

### Navis integration (currently unavailable)

> **Current release status:** the Navis bridge is known to be broken and is
> not supported in this release. It is unrelated to Prism or Argus paper
> reproduction. The configuration below is retained as interface reference
> only; do not rely on it for an operational export workflow.

You can configure Navis directly in the UI at:
- `http://localhost:8000/static/settings.html`

If you run Navis on localhost and want Prism to export each run:

```bash
export PRISM_NAVIS_URL="http://127.0.0.1:9000"
export PRISM_NAVIS_HEALTH_PATH="/health"          # optional, default: /health
export PRISM_NAVIS_EXPORT_PATH="/api/prism/runs"  # optional, default: /api/prism/runs
export PRISM_NAVIS_TIMEOUT_S="0.35"               # optional probe timeout in seconds
export PRISM_NAVIS_AUTO_EXPORT="1"                # optional, auto-send after successful run
python src/server.py
```

Runtime endpoints exposed by Prism:
- `GET /api/navis/status` checks whether Navis is reachable.
- `POST /api/navis/export` with `{ "run_id": "..." }` exports one run on demand.

Detailed Navis-side receiver and payload documentation:
- [docs/NAVIS_INTEGRATION.md](docs/NAVIS_INTEGRATION.md)

## Inputs

Prism expects:
- A DASH MPD and encoded component-tiled V3C content (typically prepared with V3CTK; see below)
- A directory of original PLY frames (for frame index inference)
- A CSV network trace
- A CSV FoV trace

For traces, you can use the default public datasets or your own:

- Default traces (auto-download helpers):
```bash
python src/scripts/get_network_traces.py
python src/scripts/get_fov_traces.py
```

- Exact Argus paper inputs (two MPDs, 5,400 PLY frames, and the 27 evaluated
  MazeLab traces) are fetched from the published Zenodo record:
```bash
python src/scripts/get_argus_mmsys2027_data.py
```
This verifies every downloaded archive, installs the files below `data/`, and
registers the `s7` and `s15` dataset profiles used by the paper batch plans.

- Your own traces:
Place custom CSV files under `data/network_traces/` and `data/fov_traces/`, or provide explicit trace paths in run configuration/API payloads.

### Custom Trace Format (Important)

#### Network trace CSV
- Must be a comma-separated CSV with a header row and exactly 2 numeric columns.
- Column 1: time in seconds.
- Column 2: bandwidth in Mbps.
- Time must be monotonic (increasing).
- Recommended: start time at `1` (the current loader internally shifts time by `-1` second for compatibility with the default dataset).

Example:
```csv
Time,Bandwidth
1,25.0
2,24.1
3,18.7
4,20.2
```

#### FoV trace CSV
- Must include `FrameNumber` (used as the index).
- For compatibility with Prism predictors, include:
  - `HMDPX`, `HMDPY`, `HMDPZ` (position)
  - `HMDRX`, `HMDRY`, `HMDRZ` (rotation in degrees)
- `FrameNumber` should be contiguous and start at `0` for a clip/loop.
- `FrameNumber` must align with the PLY frame sequence used by the run (first PLY frame corresponds to FoV frame `0`).

Minimal example:
```csv
FrameNumber,HMDPX,HMDPY,HMDPZ,HMDRX,HMDRY,HMDRZ
0,0.12,1.65,-0.44,2.1,180.0,0.0
1,0.14,1.66,-0.43,2.4,181.2,0.1
2,0.15,1.66,-0.42,2.6,182.0,0.1
```

## Preparing MPD + Encoded Content (V3CTK)

Prism consumes DASH MPDs and encoded V3C assets. To prepare those assets, use V3CTK (the companion end-to-end content preparation toolkit):

Ouellette, J., Sidhu, J. S., and Bentaleb, A. 2026. *V3CTK: An End-to-End V3C Content Preparation Toolkit for Tiled Dynamic Point Cloud Streaming.* In Proceedings of the ACM Multimedia Systems Conference 2026 (MMSys '26), April 4-8, 2026, Hong Kong, Hong Kong. ACM, New York, NY, USA, 6 pages. https://doi.org/10.1145/3793853.3799816

## Web Frontend (Optional)

The frontend lives in `src/web/` (Svelte + TypeScript):

```bash
cd src/web
npm ci
npm run build
```

The Python server serves `src/web/dist` at `/static/*`.
If `dist` is missing, the server reports the explicit build command instead of installing dependencies automatically.

## Reports

The Reports page validates named experiment cohorts and creates frozen,
versioned input snapshots for publication outputs. See [docs/reporting.md](docs/reporting.md).

For the forthcoming Argus paper, use the dedicated
[reproduction guide](docs/argus_mmsys2027/README.md). It covers the Zenodo data package,
released learned models, paper plans, and report regeneration.

## Repository Layout

- `src/`: simulator source code
- `src/web/`: web UI source
- `src/scripts/`: trace preparation scripts
- `results/`: generated run outputs (ignored by git)
- `data/`: local datasets (ignored by git)

## Licensing

Prism uses a dual-licensing model:
- Open-source use: GNU General Public License v3.0 ([LICENSE](LICENSE))
- Commercial/proprietary use: separate commercial terms ([commercial licensing](docs/legal/COMMERCIAL-LICENSE.md))

For contribution terms, see [CONTRIBUTING.md](CONTRIBUTING.md) and the [CLA](docs/legal/CLA.md).


## Citation

If you use Prism in academic work, cite:

Jeremy Ouellette and Abdelhak Bentaleb. 2026. Prism: A Trace-Driven Simulator for Component-Aware Adaptive Streaming of V3C Content. In *ACM Multimedia Systems Conference 2026 (MMSys '26)*, April 4-8, 2026, Hong Kong, Hong Kong. ACM, New York, NY, USA, 6 pages. https://doi.org/10.1145/3793853.3799810

Copy-paste BibTeX:

```bibtex
@inproceedings{ouellette2026prism,
  author    = {Ouellette, J{\'e}r{\'e}my and Bentaleb, Abdelhak},
  title     = {Prism: A Trace-Driven Simulator for Component-Aware Adaptive Streaming of V3C Content},
  booktitle = {Proceedings of the ACM Multimedia Systems Conference 2026 (MMSys '26)},
  year      = {2026},
  address   = {Hong Kong, Hong Kong},
  publisher = {ACM},
  location  = {New York, NY, USA},
  pages     = {1--6},
  doi       = {10.1145/3793853.3799810},
  url       = {https://doi.org/10.1145/3793853.3799810}
}
```
