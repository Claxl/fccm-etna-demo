# -*- coding: utf-8 -*-
"""ETNA live dashboard — FCCM demo night.

Single-page Streamlit app that drives ETNA end-to-end, streams intermediate
metric evaluations to the browser, and renders side-by-side fixed/moving/warped
panels plus a pyramid ladder, an MI curve and the converging affine matrix.
"""
from __future__ import annotations

import queue
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st

_THIS = Path(__file__).resolve().parent
if str(_THIS) not in sys.path:
    sys.path.insert(0, str(_THIS))

from etna_runner import (  # noqa: E402
    LevelEvent,
    MetricEvent,
    StatusEvent,
    detect_fpga_status,
    run_etna_async,
)
from visualization import (  # noqa: E402
    annotate,
    decompose_affine,
    draw_landmark_error,
    load_image,
    make_checkerboard,
    make_difference,
    make_fusion,
    warp_affine,
)

IMAGES_DIR = _THIS / "images"
LEVEL_COLOURS = ["#e74c3c", "#f39c12", "#3498db", "#27ae60"]
LEVEL_LABELS = ["L0 (full)", "L1 (1/2)", "L2 (1/4)", "L3 (1/8)"]
# Plotly palette mirrored as BGR tuples for OpenCV annotations.
LEVEL_COLOURS_BGR = {
    0: (60, 76, 231),    # ~ #e74c3c
    1: (18, 156, 243),   # ~ #f39c12
    2: (219, 152, 52),   # ~ #3498db
    3: (96, 174, 39),    # ~ #27ae60
}

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(page_title="ETNA Live Demo — FCCM", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    .etna-title { font-size: 2.1rem; font-weight: 700; margin-bottom: 0; }
    .etna-sub   { color: #888; margin-top: 0; font-size: 0.95rem; }
    .kpi-card   { background:#111; border:1px solid #333; border-radius:10px;
                  padding:10px 14px; text-align:center; }
    .kpi-lbl    { color:#888; font-size:0.78rem; text-transform:uppercase; letter-spacing:1px; }
    .kpi-val    { color:#fff; font-size:1.5rem; font-weight:700; }
    .badge      { display:inline-block; padding:4px 10px; border-radius:14px;
                  font-size:0.80rem; font-weight:600; }
    .badge-ok   { background:#1e5128; color:#9ad6a2; }
    .badge-warn { background:#5a4100; color:#ffd680; }
    .badge-cpu  { background:#203a43; color:#9fd0e0; }
    .ladder-cell { border-radius:6px; padding:10px 12px; margin:3px 0;
                   font-family:monospace; font-weight:700; color:#fff;
                   border:1px solid #222; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="etna-title">ETNA — Hybrid CPU/FPGA Pyramidal Image Registration</p>',
            unsafe_allow_html=True)
st.markdown('<p class="etna-sub">Live demo • FCCM demo night</p>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

def _scan_pairs() -> dict[str, dict]:
    """Return ``{pair_name: {"fixed": Path, "moving": Path, "gt": Path | None}}``.

    Two naming conventions are accepted:

    1. ``<name>_fixed.<ext>`` + ``<name>_moving.<ext>`` (+ optional ``<name>.mat``).
    2. STAR-Bench legacy ``TAGNUMa.*`` / ``TAGNUMb.*`` + ``TAGNUM.mat``
       (TAG ∈ {CS, DN, DO, IO, MO, OO, SO}).
    """
    pairs: dict[str, dict] = {}
    if not IMAGES_DIR.exists():
        return pairs

    img_exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

    # Convention 1: <name>_fixed / <name>_moving (+ optional <name>.mat).
    for fixed in sorted(IMAGES_DIR.glob("*_fixed.*")):
        if fixed.suffix.lower() not in img_exts:
            continue
        stem = fixed.stem[:-len("_fixed")]
        for ext in img_exts:
            moving = IMAGES_DIR / f"{stem}_moving{ext}"
            if moving.exists():
                gt = IMAGES_DIR / f"{stem}.mat"
                pairs[stem] = {
                    "fixed": fixed, "moving": moving,
                    "gt": gt if gt.exists() else None,
                }
                break

    # Convention 2: STAR-Bench TAGNUMa/b + TAGNUM.mat.
    sb_tags = ("CS", "DN", "DO", "IO", "MO", "OO", "SO")
    for mat in sorted(IMAGES_DIR.glob("*.mat")):
        stem = mat.stem
        if len(stem) < 3 or stem[:2].upper() not in sb_tags:
            continue
        if stem in pairs:
            continue
        moving = fixed = None
        for ext in img_exts:
            cand_m = IMAGES_DIR / f"{stem}a{ext}"
            cand_f = IMAGES_DIR / f"{stem}b{ext}"
            if cand_m.exists() and cand_f.exists():
                moving, fixed = cand_m, cand_f
                break
        if fixed and moving:
            pairs[stem] = {"fixed": fixed, "moving": moving, "gt": mat}

    return pairs


pairs = _scan_pairs()

with st.sidebar:
    st.header("Controls")

    if not pairs:
        st.error(
            "No image pairs found.\n\n"
            f"Drop `<name>_fixed.png` + `<name>_moving.png` (+ optional "
            f"`<name>.mat` with ground-truth landmarks) in "
            f"`{IMAGES_DIR.relative_to(_THIS.parent)}/`."
        )
        pair_name = None
    else:
        def _fmt(name: str) -> str:
            mark = " ⭐" if pairs[name].get("gt") else ""
            return f"{name}{mark}"

        pair_name = st.selectbox("Image pair", list(pairs.keys()),
                                 index=0, format_func=_fmt)
        if pair_name is not None:
            if pairs[pair_name].get("gt"):
                st.caption(f"⭐ ground truth: `{pairs[pair_name]['gt'].name}`")
            else:
                st.caption("no .mat found — MI-only mode")

    device = st.radio("Device", ["CPU", "FPGA"], horizontal=True,
                      help="FPGA silently falls back to CPU if PYNQ/overlay is not available.")
    metric = st.selectbox("Similarity metric", ["mi", "mse", "cc"], index=0)
    optimizer = st.selectbox("Optimizer", ["powell", "oneplusone"], index=0)
    ref_size = st.select_slider("Reference size", [128, 256, 512], value=256)

    run_clicked = st.button("Run ETNA", type="primary", use_container_width=True,
                            disabled=pair_name is None)
    reset_clicked = st.button("Reset", use_container_width=True)

    st.markdown("---")
    fpga_probe_active, fpga_probe_msg = detect_fpga_status(device == "FPGA")
    if device == "FPGA":
        badge_cls = "badge-ok" if fpga_probe_active else "badge-warn"
    else:
        badge_cls = "badge-cpu"
    st.markdown(f'<span class="badge {badge_cls}">{fpga_probe_msg}</span>',
                unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "runs" not in st.session_state:
    st.session_state.runs = {}  # key -> RunResult-like dict, for CPU/FPGA speedup comparison

if reset_clicked:
    st.session_state.clear()
    st.rerun()

# ---------------------------------------------------------------------------
# Layout — live panels
# ---------------------------------------------------------------------------

if pair_name is None:
    st.info(
        f"Place your satellite / multimodal image pairs under **{IMAGES_DIR}** "
        "using the naming convention `<pair>_fixed.png` + `<pair>_moving.png`."
    )
    st.stop()

pair_info = pairs[pair_name]
fixed_path = pair_info["fixed"]
moving_path = pair_info["moving"]
gt_path = pair_info.get("gt")
fixed_img = load_image(fixed_path, target_size=ref_size)
moving_img = load_image(moving_path, target_size=ref_size)

st.subheader("Inputs & live overlay")
top_cols = st.columns(3)
with top_cols[0]:
    st.markdown("**Fixed**")
    fixed_slot = st.empty()
    fixed_slot.image(annotate(fixed_img, f"Fixed - {fixed_path.name}", (180, 180, 180)),
                     channels="BGR", use_container_width=True)
with top_cols[1]:
    st.markdown("**Moving**")
    moving_slot = st.empty()
    moving_slot.image(annotate(moving_img, f"Moving - {moving_path.name}", (0, 128, 255)),
                      channels="BGR", use_container_width=True)
with top_cols[2]:
    st.markdown("**Live overlay**")
    overlay_slot = st.empty()
    overlay_slot.image(
        annotate(make_fusion(fixed_img, moving_img), "initial overlay", (0, 0, 255)),
        channels="BGR", use_container_width=True,
    )

st.markdown("---")
st.subheader("Convergence telemetry")
mid_cols = st.columns([1, 2, 2])

with mid_cols[0]:
    st.markdown("**Pyramid ladder**")
    ladder_slot = st.empty()
    st.markdown("**Transform (2×3)**")
    matrix_slot = st.empty()
with mid_cols[1]:
    st.markdown("**Similarity metric — live**")
    curve_slot = st.empty()
with mid_cols[2]:
    st.markdown("**Landmark RMSE vs ground truth — live**")
    rmse_slot = st.empty()

st.markdown("---")
kpi_cols = st.columns(6)
backend_slot = kpi_cols[0].empty()
level_slot = kpi_cols[1].empty()
eval_slot = kpi_cols[2].empty()
time_slot = kpi_cols[3].empty()
rate_slot = kpi_cols[4].empty()
rmse_kpi_slot = kpi_cols[5].empty()


def render_ladder(active_level: int) -> str:
    cells = []
    for idx, label in enumerate(LEVEL_LABELS):
        is_active = (idx == active_level)
        bg = LEVEL_COLOURS[idx] if is_active else "#2a2a2a"
        border = "#ffffff" if is_active else "#333"
        size = {0: "1×", 1: "½×", 2: "¼×", 3: "⅛×"}[idx]
        cells.append(
            f'<div class="ladder-cell" style="background:{bg};border-color:{border};">'
            f"{label} &nbsp; <span style='opacity:.8;font-size:0.78rem'>{size}</span></div>"
        )
    return "".join(cells)


def render_matrix(h: np.ndarray | None) -> str:
    if h is None:
        return "<pre style='color:#888'>— waiting —</pre>"
    rows = []
    for i in range(h.shape[0]):
        cells = "  ".join(f"{h[i, j]:+7.3f}" for j in range(h.shape[1]))
        rows.append(cells)
    body = "\n".join(rows)
    return f"<pre style='color:#fff;font-size:1.05rem'>{body}</pre>"


def render_kpi(slot, label: str, value: str, color: str = "#fff") -> None:
    slot.markdown(
        f'<div class="kpi-card"><div class="kpi-lbl">{label}</div>'
        f'<div class="kpi-val" style="color:{color}">{value}</div></div>',
        unsafe_allow_html=True,
    )


# Initial state for the live slots.
ladder_slot.markdown(render_ladder(-1), unsafe_allow_html=True)
matrix_slot.markdown(render_matrix(np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)),
                     unsafe_allow_html=True)

_init_fig = go.Figure()
_init_fig.update_layout(
    template="plotly_dark", height=300, margin=dict(l=10, r=10, t=20, b=30),
    xaxis_title="metric eval #", yaxis_title="similarity value",
    showlegend=True, legend=dict(orientation="h", y=-0.2),
)
curve_slot.plotly_chart(_init_fig, use_container_width=True, key="curve-init")

_init_rmse = go.Figure()
_init_rmse.update_layout(
    template="plotly_dark", height=300, margin=dict(l=10, r=10, t=20, b=30),
    xaxis_title="metric eval #", yaxis_title="RMSE (px)",
    showlegend=True, legend=dict(orientation="h", y=-0.2),
)
if gt_path is None:
    _init_rmse.add_annotation(
        text="no .mat ground truth — drop <name>.mat next to the pair",
        xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        font=dict(color="#888"),
    )
rmse_slot.plotly_chart(_init_rmse, use_container_width=True, key="rmse-init")

render_kpi(backend_slot, "Backend",
           "FPGA" if (device == "FPGA" and fpga_probe_active) else ("CPU" if device == "CPU" else "CPU fallback"),
           "#27ae60" if (device == "FPGA" and fpga_probe_active) else "#3498db")
render_kpi(level_slot, "Level", "—")
render_kpi(eval_slot, "Metric evals", "0")
render_kpi(time_slot, "Elapsed", "0.0 s")
render_kpi(rate_slot, "Evals / s", "—")
render_kpi(rmse_kpi_slot, "RMSE (px)", "— / —")


# ---------------------------------------------------------------------------
# Run ETNA and stream events
# ---------------------------------------------------------------------------

def _run_and_stream():
    # Per-level bookkeeping fed by the event stream.
    series: dict[int, list[tuple[int, float]]] = {0: [], 1: [], 2: [], 3: []}
    rmse_series: dict[int, list[tuple[int, float]]] = {0: [], 1: [], 2: [], 3: []}
    level_timings: dict[int, float] = {}
    level_start_time: dict[int, float] = {}
    active_level = -1
    last_transform = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    ui_last_refresh = 0.0
    overlay_refresh_every = 6  # update fused overlay every Nth evaluation
    total_evals = 0
    run_start = time.time()
    initial_rmse: float | None = None
    best_rmse: float = float("inf")

    event_q: queue.Queue = queue.Queue()
    thread, slot = run_etna_async(
        fixed_img, moving_img,
        device="fpga" if device == "FPGA" else "cpu",
        metric=metric, optimizer=optimizer, ref_size=ref_size,
        event_queue=event_q,
        gt_mat_path=str(gt_path) if gt_path is not None else None,
    )

    status_msg = st.empty()
    status_msg.info("Launching ETNA…")

    while thread.is_alive() or not event_q.empty():
        try:
            evt = event_q.get(timeout=0.1)
        except queue.Empty:
            continue

        now = time.time()

        if isinstance(evt, StatusEvent):
            if evt.severity == "warning":
                status_msg.warning(evt.message)
            elif evt.severity == "error":
                status_msg.error(evt.message)
            elif evt.severity == "done":
                status_msg.success(f"Done in {evt.payload.get('total_time', 0):.2f}s — "
                                   f"{evt.payload.get('eval_count', 0)} metric evaluations")
                if "backend" in evt.payload:
                    render_kpi(
                        backend_slot, "Backend", evt.payload["backend"],
                        "#27ae60" if evt.payload.get("fpga_active") else "#3498db",
                    )
                final_rmse = evt.payload.get("final_rmse_px")
                if initial_rmse is not None and final_rmse is not None:
                    render_kpi(
                        rmse_kpi_slot, "RMSE (px)",
                        f"{initial_rmse:.1f} → {final_rmse:.2f}",
                        "#27ae60" if final_rmse < initial_rmse else "#e74c3c",
                    )
            else:
                status_msg.info(evt.message)
                if "backend" in evt.payload:
                    render_kpi(
                        backend_slot, "Backend", evt.payload["backend"],
                        "#27ae60" if evt.payload.get("fpga_active") else "#3498db",
                    )
                if "initial_rmse_px" in evt.payload:
                    initial_rmse = float(evt.payload["initial_rmse_px"])
                    render_kpi(rmse_kpi_slot, "RMSE (px)",
                               f"{initial_rmse:.1f} → …", "#f39c12")

        elif isinstance(evt, LevelEvent):
            if evt.phase == "start":
                # close previous level timing, open new
                if active_level >= 0 and active_level in level_start_time:
                    level_timings[active_level] = now - level_start_time[active_level]
                active_level = evt.level
                level_start_time[active_level] = now
                ladder_slot.markdown(render_ladder(active_level), unsafe_allow_html=True)
                render_kpi(level_slot, "Level", f"L{evt.level} @ {evt.level_size}px")

        elif isinstance(evt, MetricEvent):
            total_evals += 1
            lvl = max(0, min(3, evt.level))
            series[lvl].append((total_evals, evt.value))
            if evt.rmse_px is not None and not (evt.rmse_px != evt.rmse_px):  # NaN check
                rmse_series[lvl].append((total_evals, float(evt.rmse_px)))
                if evt.rmse_px < best_rmse:
                    best_rmse = float(evt.rmse_px)
            if evt.transform is not None:
                last_transform = evt.transform

            # Throttle UI refresh to ~10 fps to keep the browser happy.
            if now - ui_last_refresh > 0.1:
                ui_last_refresh = now

                # Live overlay (expensive) — only every Nth eval.
                if total_evals % overlay_refresh_every == 0:
                    warped_live = warp_affine(moving_img, last_transform,
                                              out_size=(ref_size, ref_size))
                    overlay_slot.image(
                        annotate(make_fusion(fixed_img, warped_live),
                                 f"live overlay - L{lvl} | eval {total_evals}",
                                 LEVEL_COLOURS_BGR[lvl]),
                        channels="BGR", use_container_width=True,
                    )

                # MI / metric curve
                fig = go.Figure()
                for idx, pts in series.items():
                    if not pts:
                        continue
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    fig.add_trace(go.Scatter(
                        x=xs, y=ys, mode="lines+markers",
                        name=LEVEL_LABELS[idx],
                        line=dict(color=LEVEL_COLOURS[idx], width=2),
                        marker=dict(size=4),
                    ))
                fig.update_layout(
                    template="plotly_dark", height=300,
                    margin=dict(l=10, r=10, t=20, b=30),
                    xaxis_title="metric eval #", yaxis_title="similarity value",
                    showlegend=True, legend=dict(orientation="h", y=-0.2),
                )
                curve_slot.plotly_chart(fig, use_container_width=True,
                                        key=f"curve-{total_evals}")

                # Landmark RMSE curve (ground truth).
                rmse_fig = go.Figure()
                if any(rmse_series.values()):
                    for idx, pts in rmse_series.items():
                        if not pts:
                            continue
                        xs = [p[0] for p in pts]
                        ys = [p[1] for p in pts]
                        rmse_fig.add_trace(go.Scatter(
                            x=xs, y=ys, mode="lines+markers",
                            name=LEVEL_LABELS[idx],
                            line=dict(color=LEVEL_COLOURS[idx], width=2),
                            marker=dict(size=4),
                        ))
                    if initial_rmse is not None:
                        rmse_fig.add_hline(
                            y=initial_rmse, line_dash="dash",
                            line_color="#888",
                            annotation_text=f"initial = {initial_rmse:.1f}px",
                            annotation_position="top right",
                        )
                else:
                    rmse_fig.add_annotation(
                        text="no ground truth loaded for this pair",
                        xref="paper", yref="paper", x=0.5, y=0.5,
                        showarrow=False, font=dict(color="#888"),
                    )
                rmse_fig.update_layout(
                    template="plotly_dark", height=300,
                    margin=dict(l=10, r=10, t=20, b=30),
                    xaxis_title="metric eval #", yaxis_title="RMSE (px)",
                    showlegend=True, legend=dict(orientation="h", y=-0.2),
                )
                rmse_slot.plotly_chart(rmse_fig, use_container_width=True,
                                       key=f"rmse-{total_evals}")

                # Transform matrix
                matrix_slot.markdown(render_matrix(last_transform), unsafe_allow_html=True)

                # KPIs
                elapsed = now - run_start
                render_kpi(eval_slot, "Metric evals", f"{total_evals}")
                render_kpi(time_slot, "Elapsed", f"{elapsed:.2f} s")
                render_kpi(rate_slot, "Evals / s",
                           f"{total_evals / max(elapsed, 1e-3):.1f}")
                if initial_rmse is not None and best_rmse != float("inf"):
                    delta_colour = "#27ae60" if best_rmse < initial_rmse else "#e74c3c"
                    render_kpi(rmse_kpi_slot, "RMSE (px)",
                               f"{initial_rmse:.1f} → {best_rmse:.2f}",
                               delta_colour)

    thread.join(timeout=2.0)

    if "error" in slot:
        st.error(f"Run failed: {slot['error']}")
        return None
    return slot.get("result")


if run_clicked:
    result = _run_and_stream()

    if result is not None:
        # Cache this run for later CPU-vs-FPGA speedup comparison.
        st.session_state.runs.setdefault(pair_name, {})[result.backend] = {
            "total_time": result.total_time,
            "eval_count": result.eval_count,
        }

        st.markdown("---")
        st.subheader("Results")
        tab_labels = ["Overlays", "Metrics", "Transform"]
        gt_available = (result.landmarks_mov_scaled is not None
                        and result.gt_transform_ref is not None)
        if gt_available:
            tab_labels.append("Ground truth")
        tabs = st.tabs(tab_labels)

        with tabs[0]:
            cc = st.columns(4)
            cc[0].image(annotate(make_fusion(result.fixed, result.moving), "before"),
                        channels="BGR", caption="Before — fused fixed ⊕ moving",
                        use_container_width=True)
            cc[1].image(annotate(make_fusion(result.fixed, result.warped), "after"),
                        channels="BGR", caption="After — fused fixed ⊕ warped",
                        use_container_width=True)
            cc[2].image(annotate(make_checkerboard(result.fixed, result.warped), "checker"),
                        channels="BGR", caption="Checkerboard 32×32",
                        use_container_width=True)
            cc[3].image(annotate(make_difference(result.fixed, result.warped), "|diff|"),
                        channels="BGR", caption="Absolute diff (JET)",
                        use_container_width=True)

        with tabs[1]:
            mc = st.columns(3)
            mc[0].metric("Total time", f"{result.total_time:.3f} s")
            mc[1].metric("Metric evaluations", f"{result.eval_count}")
            mc[2].metric("Backend", result.backend)

            if result.initial_rmse_px is not None and result.final_rmse_px is not None:
                rc = st.columns(3)
                rc[0].metric("Initial RMSE", f"{result.initial_rmse_px:.2f} px")
                rc[1].metric("Final RMSE", f"{result.final_rmse_px:.2f} px",
                             delta=f"{result.final_rmse_px - result.initial_rmse_px:+.2f}",
                             delta_color="inverse")
                improvement = (1.0 - result.final_rmse_px / max(result.initial_rmse_px, 1e-6)) * 100
                rc[2].metric("Improvement", f"{improvement:.1f} %")

            # CPU-vs-FPGA speedup panel (only if both backends have been run).
            cached = st.session_state.runs.get(pair_name, {})
            cpu_entry = next((v for k, v in cached.items() if "CPU" in k), None)
            fpga_entry = next((v for k, v in cached.items() if "FPGA" in k and "fallback" not in k), None)
            if cpu_entry and fpga_entry:
                speedup = cpu_entry["total_time"] / max(fpga_entry["total_time"], 1e-6)
                st.success(f"FPGA speedup vs CPU on **{pair_name}**: "
                           f"**×{speedup:.2f}** "
                           f"({cpu_entry['total_time']:.2f}s → {fpga_entry['total_time']:.2f}s)")
            else:
                st.caption("Run the same pair on both CPU and FPGA to see the speedup here.")

        with tabs[2]:
            H = result.transform
            st.code(
                f"Final 2x3 affine matrix:\n"
                f"  [ {H[0,0]:+8.4f}  {H[0,1]:+8.4f}  {H[0,2]:+8.4f} ]\n"
                f"  [ {H[1,0]:+8.4f}  {H[1,1]:+8.4f}  {H[1,2]:+8.4f} ]",
                language="text",
            )
            decomp = decompose_affine(H)
            dc = st.columns(5)
            dc[0].metric("Δx (px)", f"{decomp['tx']:+.2f}")
            dc[1].metric("Δy (px)", f"{decomp['ty']:+.2f}")
            dc[2].metric("Rotation (°)", f"{decomp['rot_deg']:+.2f}")
            dc[3].metric("Scale X", f"{decomp['scale_x']:.3f}")
            dc[4].metric("Scale Y", f"{decomp['scale_y']:.3f}")

            if result.gt_transform is not None:
                Tg = np.asarray(result.gt_transform, dtype=np.float32)
                if Tg.shape == (3, 3):
                    Tg_23 = Tg[:2, :]
                else:
                    Tg_23 = Tg
                st.code(
                    f"Ground truth 2x3 affine:\n"
                    f"  [ {Tg_23[0,0]:+8.4f}  {Tg_23[0,1]:+8.4f}  {Tg_23[0,2]:+8.4f} ]\n"
                    f"  [ {Tg_23[1,0]:+8.4f}  {Tg_23[1,1]:+8.4f}  {Tg_23[1,2]:+8.4f} ]",
                    language="text",
                )

        if result.landmarks_mov_scaled is not None and result.gt_transform_ref is not None:
            with tabs[3]:
                # STAR-Bench landmark-error convention:
                #   predicted = H_est  @ lm_mov
                #   truth     = T_gt_ref @ lm_mov
                from etna_runner import _apply_affine_2x3 as _aff
                predicted_lm = _aff(result.landmarks_mov_scaled, result.transform)
                gt_lm_on_fixed = _aff(result.landmarks_mov_scaled, result.gt_transform_ref)
                err_img = draw_landmark_error(
                    result.fixed, gt_lm_on_fixed, predicted_lm,
                )
                c1, c2 = st.columns([2, 1])
                with c1:
                    st.image(annotate(err_img,
                                      f"GT (green) vs predicted (red) - RMSE {result.final_rmse_px:.2f} px",
                                      (0, 255, 0)),
                             channels="BGR", use_container_width=True)
                with c2:
                    st.metric("Landmarks", f"{len(result.landmarks_mov_scaled)}")
                    st.metric("Initial RMSE",
                              f"{result.initial_rmse_px:.2f} px"
                              if result.initial_rmse_px is not None else "—")
                    st.metric("Final RMSE",
                              f"{result.final_rmse_px:.2f} px"
                              if result.final_rmse_px is not None else "—")
                    # Per-landmark error histogram: ||H @ p - T_gt @ p||.
                    per_lm = np.linalg.norm(predicted_lm - gt_lm_on_fixed, axis=1)
                    hist_fig = go.Figure(data=[go.Histogram(x=per_lm, nbinsx=20,
                                                            marker_color="#3498db")])
                    hist_fig.update_layout(
                        template="plotly_dark", height=220,
                        margin=dict(l=10, r=10, t=20, b=30),
                        xaxis_title="per-landmark error (px)",
                        yaxis_title="count",
                    )
                    st.plotly_chart(hist_fig, use_container_width=True,
                                    key="per-lm-hist")
else:
    st.caption("Pick a pair in the sidebar and hit **Run ETNA** to start the live demo.")
