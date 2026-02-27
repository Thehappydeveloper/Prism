# Prism: V3C Component-Tiled Streaming Simulator (MMSys 2026)

Prism is an offline, trace-driven simulator for adaptive streaming of MPEG-I V3C / V-PCC content described by DASH-aligned MPDs.
It was developed as a research artifact for the MMSys 2026 Prism paper, and is intended for researchers and engineers who want to evaluate ABR logic, FoV prediction behavior, and QoE tradeoffs before deploying changes in production players.
Prism models viewport-aware and component-aware adaptation over spatially tiled V3C content, replays network and FoV traces, and writes run artifacts to `results/`.
It simulates the DASH control loop without decoding or rendering, so experiments stay reproducible and fast to iterate.

## Quick Start

### Requirements
- Python 3.10+
- Node.js 18+ (required on first run if `src/web/dist` is missing)

### Platform Recommendations
- Developed and tested primarily on NixOS Linux.
- Linux is the recommended runtime environment.
- On Windows, use WSL2 or Docker for the most reliable setup and dependency behavior.

### Install Python dependencies
```bash
python -m pip install -r requirements.txt
```

### Run the server
```bash
python src/server.py
```

Open `http://localhost:8000/`.

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
npm install
npm run build
```

The Python server serves `src/web/dist` at `/static/*`.
If `dist` is missing, the server will try to run `npm install && npm run build` automatically.

## Repository Layout

- `src/`: simulator source code
- `src/web/`: web UI source
- `src/scripts/`: trace preparation scripts
- `results/`: generated run outputs (ignored by git)
- `data/`: local datasets (ignored by git)

## Licensing

Prism uses a dual-licensing model:
- Open-source use: GNU General Public License v3.0 ([LICENSE](LICENSE))
- Commercial/proprietary use: separate commercial terms ([COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md))

For contribution terms, see [CONTRIBUTING.md](CONTRIBUTING.md) and [CLA.md](CLA.md).


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
