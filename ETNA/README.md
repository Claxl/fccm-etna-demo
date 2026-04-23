# ETNA

Hybrid CPU/FPGA **image registration bundle** shipped with STAR-Bench. ETNA
packages three things behind a single Python import:

- **`ETNA_Multi`** — the pyramidal (multi-resolution) Mutual-Information
  engine, with its own metric component (`EtnaMultiMetric`) and two
  optimizers (`EtnaMultiPowell`, `EtnaMultiOnePlusOne`).
- **`PipelineExecutor`** — a multi-stage runner that chains a feature-based
  front-end (e.g. XFeat) with `ETNA_Multi` refinement, composing the
  individual transforms into a single final affine. This is how ETNA exposes
  "XFeat → ETNA_Multi" and similar hybrid pipelines.
- The **`wax_mi_accel`** FPGA accelerator overlay for Xilinx Kria boards, via
  a singleton PYNQ driver (`FaberFPGAAccelerator`).

STAR-Bench itself only runs **one detector per evaluation entry**. If you
want to evaluate a hybrid chain (e.g. XFeat feeding into ETNA_Multi Powell),
invoke `ETNA.PipelineExecutor` directly from Python — it intentionally stays
out of the STAR-Bench harness.

The non-pyramidal Faber flavour lives outside ETNA: it is pulled as a
submodule from `github.com/necst/faber_fpga` and consumed via
`wrappers/faber_wrapper.py` at the STAR-Bench level.

This directory is structured as a self-contained Python package so it can be
consumed either as a subfolder of `STAR-Bench` or as a standalone git
submodule from its own repository.

## Authors

Claudio Di Salvo, Emanuele Del Sozzo, Giuseppe Sorrentino, Eleonora D'Arnese,
Paolo Panicucci, Davide Conficconi.

## Public API

```python
from ETNA import (
    # Hyper-params
    FaberHyperParams, FaberOneplusoneCommonFunctions,
    # FPGA singleton driver
    FaberFPGAAccelerator,
    # ETNA_Multi (pyramidal) metric + optimizers
    EtnaMultiMetric,
    EtnaMultiPowell,
    EtnaMultiOnePlusOne,
    AdaptiveParameters, ImagePyramid,
    # Advanced: multi-stage pipeline runner (ETNA_Multi + XFeat combinations)
    PipelineExecutor, StageCache,
)
```

`PipelineExecutor` and `StageCache` are **advanced, Python-only APIs**:
they are never constructed by the STAR-Bench CLI (`starbench-main.py`
always runs a single backend per entry). Import them directly when you
need to chain a feature detector and an intensity refiner in the same
process; drop the `cache=` argument when registering a single pair, as
`StageCache` is just an in-process memoiser for heavy detectors.

## Files

| File                         | Purpose                                                  |
|------------------------------|----------------------------------------------------------|
| `__init__.py`                | Package re-exports                                       |
| `hyperparams.py`             | `FaberHyperParams`, Marsaglia normal-variate generator   |
| `fpga_accelerator.py`        | Singleton PYNQ overlay driver (AXI-Lite + uncached DMA)  |
| `registrators_pyramidal.py`  | `EtnaMultiMetric` (MI, MSE, CC, Parzen)                  |
| `optimizers_pyramidal.py`    | `EtnaMultiPowell`, `EtnaMultiOnePlusOne`                 |
| `pipeline.py`                | `PipelineExecutor`, `StageCache` — multi-stage chains    |
| `etna.bit`, `etna.hwh`       | Xilinx Kria bitstream / hardware handoff                 |

## Minimal usage

```python
from ETNA import EtnaMultiMetric, EtnaMultiPowell
import cv2, torch

ref = torch.tensor(cv2.imread("fixed.png", 0), dtype=torch.uint8)
flt = torch.tensor(cv2.imread("moving.png", 0), dtype=torch.uint8)

metric = EtnaMultiMetric(ref_size=256, metric="mi", exponential=True, use_fpga=False)
optimizer = EtnaMultiPowell()
H = optimizer.compute(ref, flt, metric_component=metric, image_dimension=256)
```

### Running an ETNA multi-stage pipeline

```python
from ETNA import PipelineExecutor, StageCache
from starbench_detectors import DetectorFactory

exec_ = PipelineExecutor(
    pipeline_config="xfeat,etna_multi_powell_mi_fpga",
    detector_factory=DetectorFactory(),
    device="cpu",
    cache=StageCache(),
)
result = exec_.run("fixed.png", "moving.png")
print(result["final_transform"])
```

Set `use_fpga=True` to offload the MI kernel to the `wax_mi_accel` overlay.
If initialisation fails (for example because PYNQ isn't installed or the
bitstream can't be loaded), the metric transparently falls back to the
software CUDA/CPU implementation.


## Notes on the standalone driver

A legacy standalone driver for ETNA SW is kept at
`../fabersw-unique.py`. It glob-scans a folder for `*a.png` / `*b.png`
pairs and runs the pyramidal optimiser — useful for quick smoke tests
outside the full STAR-Bench pipeline.

## License

MIT License — see the header of each Python file. Portions derive from the
original Faber release by Eleonora D'Arnese, Davide Conficconi, Emanuele Del
Sozzo, Luigi Fusco, Donatella Sciuto, and Marco Domenico Santambrogio (2022).
