# ETNA Live Demo — FCCM Demo Night

Single-page Streamlit dashboard that drives the pyramidal **ETNA_Multi**
image-registration engine end-to-end and streams its intermediate state to
the browser in real time: fixed / moving / live overlay panels, pyramid
ladder, metric curve, converging affine matrix, per-level timings, and a
CPU-vs-FPGA speedup panel.

## What it shows

- **Coarse-to-fine pyramid in action** — 4 pyramid levels (L3 → L0), the
  active one is highlighted on the ladder as the optimizer descends.
- **Live overlay** — the moving image is warped with the current transform
  every few metric evaluations and fused 50/50 with the fixed image so you
  *see* the registration converge.
- **Metric curve** — one Plotly line per pyramid level, updated live.
- **Landmark RMSE vs ground truth** — when a `.mat` GT file ships next to
  the pair, a second live curve shows the pixel-space error at every
  metric evaluation (the distance from the golden truth) and the header
  KPI ticks from `initial RMSE` down to `final RMSE`.
- **Affine matrix** — the 2×3 transform ticks toward its final value.
- **Backend badge** — `CPU`, `FPGA: wax_mi_accel active`, or
  `FPGA requested -> software fallback` (on laptops without the Kria board).
- **Results tabs** — Overlays (before / after / checkerboard / diff),
  Metrics (with CPU-vs-FPGA speedup once the same pair has run on both),
  Transform (decomposition + GT comparison), and Ground truth (GT vs
  predicted landmark overlay with per-landmark error histogram).

## Requirements

- Python 3.10+.
- Every runtime dependency is pinned in `requirements-demo.txt` —
  this folder is **self-contained**, you do not need the rest of the
  STAR-Bench repo.
- **Optional:** `pynq` (uncomment in `requirements-demo.txt`) on a
  Xilinx Kria KV260 / KR260 to enable the FPGA MI path. Missing PYNQ is
  fine on a laptop: the demo falls back to the software MI implementation.

## Setup

```bash
cd DEMO-FCCM
pip install -r requirements-demo.txt
```

That single `pip install` pulls `numpy`, `scipy`, `opencv-python`, `torch`,
`torchvision`, `kornia`, `streamlit` and `plotly` — everything the demo
needs end-to-end.

## Run

```bash
cd DEMO-FCCM
./run_demo.sh                  # http://localhost:8501
PORT=8600 ./run_demo.sh        # pick a different port
```

Or directly:

```bash
streamlit run app.py
```

## Add image pairs

Drop the demo pairs in `DEMO-FCCM/images/` using either:

```
<name>_fixed.png
<name>_moving.png
<name>.mat          # optional — STAR-Bench ground truth
```

or the STAR-Bench legacy naming `TAGNUMa.*` / `TAGNUMb.*` + `TAGNUM.mat`.

Both `.png`, `.jpg`, `.tif`, `.bmp` are accepted. Any dimensions work —
images are resized to the sidebar's `Reference size` and landmarks are
rescaled to match. See `images/README.md` for the full `.mat` schema.

## Flow

1. Pick a pair in the sidebar.
2. Pick **Device: CPU** or **FPGA**. (FPGA falls back to CPU automatically
   when PYNQ isn't available — no crash, just a yellow badge.)
3. Pick metric (`mi` / `mse` / `cc`), optimizer (`powell` / `oneplusone`)
   and reference size (`128` / `256` / `512`).
4. Hit **Run ETNA** and watch the dashboard update live.
5. After the run finishes, flip to the **Results / Metrics / Transform**
   tabs for the full breakdown.
6. Re-run the same pair with the other device to see the speedup.

## Headless smoke test

```bash
cd DEMO-FCCM
python etna_runner.py images/<name>_fixed.png images/<name>_moving.png \
    --device cpu --metric mi --optimizer powell --ref-size 256 \
    --gt images/<name>.mat                              # optional
```

## Files

```
DEMO-FCCM/                  # fully self-contained demo bundle
├── app.py                 # Streamlit dashboard
├── etna_runner.py         # Instrumented ETNA wrapper + threaded runner
├── visualization.py       # fusion / checkerboard / difference helpers
├── gt_loader.py           # standalone .mat ground-truth loader
├── ETNA/                  # pyramidal MI engine (moved in-tree)
│   ├── __init__.py
│   ├── registrators_pyramidal.py
│   ├── optimizers_pyramidal.py
│   ├── fpga_accelerator.py
│   ├── hyperparams.py
│   ├── pipeline.py
│   ├── etna.bit           # Xilinx Kria bitstream
│   └── etna.hwh
├── requirements-demo.txt  # streamlit, plotly
├── run_demo.sh            # launcher
├── images/                # drop your pairs (+ optional .mat) here
└── README.md
```

The ETNA package is shipped **inside** `DEMO-FCCM/` so the demo can be
zipped and run on any machine (laptop or Kria board) without pulling the
rest of the STAR-Bench repo. Nothing inside `ETNA/` is patched: live hooks
are attached via a transparent subclass of `EtnaMultiMetric` defined in
`etna_runner.py`.
