# Argus MMSys 2027 reproduction

Use this guide only to reproduce the Argus paper results. For custom datasets
or general Prism use, see the [repository README](../../README.md).

Choose **one** path below. The recommended path regenerates the submitted
tables and figures from bundled artifacts; it does not rerun simulations. The
full re-simulation path is only for reviewers who want to execute every paper
experiment again.

## 1. Install the shared prerequisites — once

Both paths use Python 3.13 and the base locked environment. Conda is the
recommended setup because it installs the requested interpreter independently
of the system Python (including on NixOS):

```bash
conda create -n prism-argus python=3.13 pip
conda activate prism-argus
python -m pip install --upgrade pip
python -m pip install -r requirements.lock
```

If you prefer `venv`, create it explicitly with Python 3.13 rather than the
system default:

```bash
# Linux/macOS/NixOS:
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.lock
```

On Windows PowerShell, use:

```powershell
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.lock
```

Install `ffmpeg` and ImageMagick with your operating system's package manager.
They are needed only for the tile-score comparison figure, but that figure is
part of the complete parent report. Node.js is **not** needed for the
command-line workflow; install Node.js 18+ only if you want the browser UI.

## 2A. Recommended: regenerate reports from bundled runs

The repository contains the 2,219 completed run artifacts required by the
Argus report (about 1.59 GiB). Run this command once:

```bash
python src/scripts/reproduce_argus_mmsys2027.py --precomputed
```

It checks the Python packages, `ffmpeg`, ImageMagick, the bundled manifest,
and exactly 2,219 finished runs. It then builds the complete parent report.
Expect a few minutes, not a simulation run. A ten-phase terminal progress bar
names the current table or figure renderer and shows elapsed time while it runs.

This path needs no Zenodo download, PyTorch, learned-model setup, dataset
profile, transform, Node.js, or saved UI settings. Tables, metrics, and
tile-score maps are reconstructed from the run artifacts. The only omitted
detail is optional point-cloud and motion-trace overlays in the tile-score
figure; obtain the Zenodo data only if you want those overlays or to inspect
the original inputs.

## 2B. Alternative: rerun every paper experiment

Do this **instead of 2A** only if you want a new simulation cohort. It runs
2,219 simulations and normally takes more than one day. Raycasting measures
real machine work, so rerun timing and adaptive outcomes can vary by host;
do not expect bit-identical timing results.

Install the additional inference dependency and fetch the paper inputs:

```bash
python -m pip install -r requirements-training.lock
python src/scripts/get_argus_mmsys2027_data.py
python src/scripts/reproduce_argus_mmsys2027.py --rerun
```

The fetcher downloads 37.2 GB and installs about 228 GiB of extracted data;
reserve at least 270 GiB free beneath `data/`. It verifies the archives and
automatically registers portable `s7` and `s15` dataset profiles. The rerun
check confirms that PyTorch, the two released MLP/LSTM models, and all data
are available.

If that check succeeds, start Prism and open the parent report:

```bash
cd src/web
npm ci
npm run build
cd ../..
python src/server.py
```

Open `http://localhost:8000/`, select **Reports**, then **Argus MMSys 2027
reproduction**. Use each **Generate** button to load one of the six JSON plans
into Batch Planner, inspect it, then click **Plan runs** and **Start batch**.
The plans are deliberately not launched automatically. The Batch Planner's
timestamp warning can be ignored: it is UI metadata and does not alter the
loaded plan or its fixed experiment settings. Repeat this for all six plans.
After all six batches finish, build the parent report. A complete newer rerun
automatically supersedes the bundled cohort; an incomplete rerun does not.
This full re-simulation path therefore requires Node.js 18+; the bundled
command-line path does not.

## 3. Expected output

Both paths produce a timestamped build at:

```text
docs/argus_mmsys2027/generated/argus_mmsys2027/<build-id>/
```

For the bundled path, success includes `2,219/2,219 finished runs`. The build
contains:

- `report-manifest.json` — report definition, selected cohort, validation, and
  output inventory;
- `inputs.json` and `runs.csv` — the frozen input snapshot and artifact
  fingerprints;
- `tables.md` — the parent input summary; and
- child-report directories for `baseline_bandwidth_qoe`,
  `master_results_full`, `raycast_process_scaling`, and
  `rank_budget_visual_comparison`, containing the detailed paper table and
  QoE, process-scaling, and tile-score figures.

If validation does not report exactly 2,219 finished runs, do not compare the
output with the paper. Restore the bundled `results/` cohort and
`results/_run_groups.json` manifest first.

See [PAPER_DATA.md](PAPER_DATA.md) for the Zenodo inventory and checksums, or
[../reporting.md](../reporting.md) for the low-level report CLI.
