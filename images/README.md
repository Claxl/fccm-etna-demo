# Demo image pairs

Drop your image pairs in this folder using either of the two conventions below.

## 1. Simple naming

```
<name>_fixed.<ext>
<name>_moving.<ext>
<name>.mat            # optional — ground-truth landmarks (see below)
```

`<ext>` ∈ `png`, `jpg`, `jpeg`, `tif`, `tiff`, `bmp`.

Examples:

```
images/
├── optical_sar_01_fixed.png
├── optical_sar_01_moving.png
├── optical_sar_01.mat       (optional)
├── visible_ir_02_fixed.tif
└── visible_ir_02_moving.tif
```

## 2. STAR-Bench legacy naming

```
TAGNUM.mat                 # ground truth (required here)
TAGNUMa.<ext>              # moving
TAGNUMb.<ext>              # fixed
```

`TAG` ∈ `CS`, `DN`, `DO`, `IO`, `MO`, `OO`, `SO`. Example: `SO001.mat`,
`SO001a.png`, `SO001b.png`.

## Ground truth `.mat` file (optional but recommended)

When a `.mat` file is found next to the pair the dashboard unlocks the
**live landmark-RMSE curve** (distance in pixels between the current
estimate and ground truth, updated at every metric evaluation), plus a
dedicated **Ground truth** tab with GT vs predicted overlay and a per-
landmark error histogram.

The expected MATLAB structure matches `starbench_utils.load_ground_truth`:

| field                             | description                                 |
|-----------------------------------|---------------------------------------------|
| `I_fix`                           | fixed image array (informational)           |
| `I_move`                          | moving image array (informational)          |
| `Landmarks.I_fix_landmark`        | Nx2 ground-truth points on the fixed image  |
| `Landmarks.I_move_landmark`       | Nx2 corresponding points on the moving image |
| `T_reg_gt`                        | 3x3 homogeneous ground-truth transform      |

Pairs without a `.mat` file still work — the dashboard falls back to
MI-only mode and tells you so.

## Notes

- The two images can have different dimensions — they are auto-resized to
  the **Reference size** picked in the sidebar (128 / 256 / 512).
  Landmarks are rescaled to the same grid automatically.
- Grayscale or RGB are both fine (RGB is converted to grayscale internally).
- Pairs that ship with a `.mat` show a ⭐ next to their name in the
  sidebar selector.
