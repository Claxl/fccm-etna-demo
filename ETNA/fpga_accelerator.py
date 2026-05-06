# -*- coding: utf-8 -*-
"""
ETNA FPGA Accelerator.

Thin wrapper over the PYNQ overlay that exposes the ``wax_mi_accel`` IP block
used to offload the Mutual-Information + affine warp kernel.

The class is implemented as a singleton so that the overlay (a very expensive
resource) is only loaded once per process even if multiple ETNA optimizers
instantiate a metric.
"""
import os
import time
import numpy as np
import torch
import cv2  # hoisted from compute_mi hot path; only used by invertAffineTransform fallback


try:
    from pynq import Overlay, allocate
    PYNQ_AVAILABLE = True
except ImportError:
    PYNQ_AVAILABLE = False
    print("WARNING: PYNQ libraries not found. FPGA acceleration will not be available.")


# Resolve the default bitstream path relative to this file so the module
# keeps working when used as a git submodule or installed elsewhere.
_DEFAULT_OVERLAY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "etna.bit")


class FaberFPGAAccelerator:
    """
    Singleton driver for the ETNA MI+Warp accelerator.

    Parameters
    ----------
    overlay_path : str, optional
        Path to the ``.bit`` file. Defaults to the bitstream shipped with the
        ETNA module.
    """

    _instance = None

    def __new__(cls, overlay_path: str = _DEFAULT_OVERLAY):
        if cls._instance is None:
            cls._instance = super(FaberFPGAAccelerator, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Drop the cached singleton so the next constructor call retries init."""
        inst = cls._instance
        if inst is not None:
            try:
                inst.close()
            except Exception:
                pass
        cls._instance = None

    def __init__(self, overlay_path: str = _DEFAULT_OVERLAY):
        # Skip re-initialization once we have a *successful* singleton. Failed
        # init attempts leave _initialized=False so the next constructor call
        # retries — important when the sidebar probe fires before the overlay
        # is ready (e.g. another process holding the device).
        if getattr(self, "_initialized", False):
            return

        self.enabled = False
        self.overlay_path = overlay_path
        self.pynq_available = PYNQ_AVAILABLE
        self.init_error: str | None = None
        self.overlay = None
        self.ip = None
        self.current_shape = (0, 0)
        self.input_flt_buffer = None
        self.input_ref_buffer = None
        self.transform_buffer = None
        self.mi_hls_buffer = None

        # Cache state: used to avoid redundant AXI-Lite register writes.
        # Signature is (data_ptr, shape, dtype) — more reliable than id() which
        # Python can reuse after a previous tensor is garbage-collected.
        self._last_ref_sig = None
        self._last_flt_sig = None
        self._registers_configured = False

        # Per-stage timing accumulators. Reset/reported by the pyramid loop
        # to identify where the per-call overhead goes.
        self._stage_times = {
            "convert": 0.0, "signature": 0.0, "copy": 0.0, "transform": 0.0,
            "regs": 0.0, "kick": 0.0, "poll": 0.0, "read": 0.0,
        }
        self._call_count = 0
        self._copy_count = 0
        self._regs_count = 0
        self._poll_reads = 0

        if not PYNQ_AVAILABLE:
            self.init_error = "PYNQ not installed (pip install pynq)"
            return

        if not os.path.exists(overlay_path):
            self.init_error = f"bitstream not found at {overlay_path}"
            print(f"ERROR: FPGA Init Failed: {self.init_error}")
            return

        try:
            # PYNQ uses asyncio for device discovery; Streamlit's ScriptRunner
            # thread has no event loop — create one if needed.
            import asyncio
            try:
                asyncio.get_event_loop()
            except RuntimeError:
                asyncio.set_event_loop(asyncio.new_event_loop())

            # On Kria, dfx-mgr / xmutil keeps the previously-loaded app firmware
            # locked. Calling `xmutil unloadapp` releases the device so we can
            # program our overlay without ETXTBSY. Best-effort: ignore errors
            # if xmutil is not installed or there is nothing to unload.
            import subprocess
            try:
                subprocess.run(
                    ["xmutil", "unloadapp"],
                    check=False, capture_output=True, timeout=5,
                )
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass

            # ETXTBSY (errno 26 "Text file busy") happens when another process
            # is still holding the bitstream. Retry up to 6 times with
            # exponential backoff (~31s total) before giving up.
            print(f"Loading FPGA Overlay: {overlay_path}...")
            _attempts = 0
            _max_attempts = 6
            while True:
                try:
                    self.overlay = Overlay(overlay_path)
                    break
                except OSError as oe:
                    if oe.errno != 26 or _attempts >= _max_attempts:
                        if oe.errno == 26:
                            # Final actionable hint for the user.
                            raise OSError(
                                oe.errno,
                                f"{oe.strerror} after {_max_attempts} retries. "
                                "Another process is holding the FPGA. Try:\n"
                                "  1) sudo xmutil unloadapp\n"
                                "  2) sudo lsof " + overlay_path + "  # find the holder\n"
                                "  3) sudo pkill -9 streamlit python  # kill stale sessions\n"
                                "  4) reboot the Kria as last resort"
                            ) from oe
                        raise
                    _attempts += 1
                    backoff = 0.5 * (2 ** _attempts)
                    print("WARNING: Overlay load got ETXTBSY, retry %d/%d in %.1fs..."
                          % (_attempts, _max_attempts, backoff))
                    time.sleep(backoff)

            # Resolve the accelerator IP block by known name or by substring.
            if hasattr(self.overlay, 'wax_mi_accel_0'):
                self.ip = self.overlay.wax_mi_accel_0
            else:
                for ip_name in self.overlay.ip_dict:
                    if 'wax' in ip_name or 'accel' in ip_name:
                        self.ip = getattr(self.overlay, ip_name)
                        break
                else:
                    raise RuntimeError(
                        f"no wax/accel IP found in overlay (have: "
                        f"{list(self.overlay.ip_dict.keys())})"
                    )

            # AXI-Lite register offsets exposed by the HLS core
            self.OFFSET_CTRL      = 0x00
            self.OFFSET_INPUT_IMG = 0x10
            self.OFFSET_INPUT_REF = 0x1c
            self.OFFSET_MI_OUT    = 0x28
            self.OFFSET_TRANSFORM = 0x34
            self.OFFSET_ROWS      = 0x40
            self.OFFSET_COLS      = 0x48

            # Uncached DMA buffers for the transform matrix and MI result.
            self.transform_buffer = allocate(shape=(16,), dtype=np.float32, cacheable=False)
            self.mi_hls_buffer = allocate(shape=(16,), dtype=np.float32, cacheable=False)

            # The bottom row of the homogeneous transform is always [0, 0, 1];
            # write it once here so compute_mi only touches indices 0..5.
            self.transform_buffer[6] = 0.0
            self.transform_buffer[7] = 0.0
            self.transform_buffer[8] = 1.0

            self.enabled = True
            self._initialized = True
            print("FPGA Accelerator Ready (Optimized Uncached Mode).")

        except Exception as e:
            self.init_error = f"{type(e).__name__}: {e}"
            print(f"ERROR: FPGA Init Failed: {self.init_error}")
            self.enabled = False

    @property
    def status_detail(self) -> str:
        """Short human-readable status string for the UI."""
        if self.enabled:
            return f"FPGA active: overlay loaded from {self.overlay_path}"
        if self.init_error:
            return f"FPGA disabled: {self.init_error}"
        return "FPGA disabled"

    @staticmethod
    def _tensor_signature(t):
        """Stable identity signature for cache invalidation.

        Returns ``(memory_pointer, shape, dtype)`` for torch tensors and
        numpy arrays. Falls back to ``id(t)`` for anything else (this only
        happens if the optimizer passes something exotic).
        """
        if isinstance(t, torch.Tensor):
            return (t.data_ptr(), tuple(t.shape), str(t.dtype))
        if isinstance(t, np.ndarray):
            return (t.ctypes.data, tuple(t.shape), str(t.dtype))
        return (id(t),)

    def _ensure_buffers(self, rows: int, cols: int):
        """Lazily (re)allocate the input image buffers if the shape changed."""
        if (rows, cols) != self.current_shape:
            if self.input_flt_buffer is not None:
                self.input_flt_buffer.freebuffer()
            if self.input_ref_buffer is not None:
                self.input_ref_buffer.freebuffer()

            # Direct-mapped RAM buffers (no CPU caching)
            self.input_flt_buffer = allocate(shape=(rows, cols), dtype=np.uint8, cacheable=False)
            self.input_ref_buffer = allocate(shape=(rows, cols), dtype=np.uint8, cacheable=False)
            self.current_shape = (rows, cols)

            # Invalidate cached register state
            self._last_ref_sig = None
            self._last_flt_sig = None
            self._registers_configured = False

    def reset_stats(self) -> None:
        """Zero the per-stage timing accumulators."""
        for k in self._stage_times:
            self._stage_times[k] = 0.0
        self._call_count = 0
        self._copy_count = 0
        self._regs_count = 0
        self._poll_reads = 0

    def report_stats(self, label: str) -> None:
        """Print a one-line per-stage summary since the last reset_stats()."""
        if self._call_count == 0:
            return
        ms = {k: v * 1000.0 for k, v in self._stage_times.items()}
        total_ms = sum(ms.values())
        print(
            f"[FPGA stats] {label} calls={self._call_count} "
            f"copy={self._copy_count} regs={self._regs_count} "
            f"polls={self._poll_reads} | "
            f"conv={ms['convert']:.1f} sig={ms['signature']:.1f} "
            f"copy={ms['copy']:.1f} xform={ms['transform']:.1f} "
            f"regs={ms['regs']:.1f} kick={ms['kick']:.1f} "
            f"poll={ms['poll']:.1f} read={ms['read']:.1f} "
            f"total={total_ms:.1f} ms"
        )

    def compute_mi(self, ref_tensor, flt_tensor, transform_matrix_2x3) -> float:
        """
        Run a single MI evaluation on the FPGA.

        This method is hot-looped by the optimizers, so it minimizes the
        amount of Python-to-AXI traffic: register writes are only issued on
        the first call or when the shape / input buffers change.
        """
        if not self.enabled:
            raise RuntimeError(self.init_error or "FPGA accelerator not enabled")

        _pc = time.perf_counter
        _stages = self._stage_times

        # 1) Compute signature first; only convert ref/flt to numpy if the
        # underlying buffer changed (otherwise no new copy is needed).
        _t = _pc()
        curr_ref_sig = self._tensor_signature(ref_tensor)
        curr_flt_sig = self._tensor_signature(flt_tensor)
        need_copy = (curr_ref_sig != self._last_ref_sig
                     or curr_flt_sig != self._last_flt_sig)
        _stages["signature"] += _pc() - _t

        # 2) Conversion (only meaningful when we will actually copy).
        _t = _pc()
        if need_copy:
            ref_np = ref_tensor.detach().cpu().numpy() if isinstance(ref_tensor, torch.Tensor) else ref_tensor
            flt_np = flt_tensor.detach().cpu().numpy() if isinstance(flt_tensor, torch.Tensor) else flt_tensor
            rows, cols = ref_np.shape
        else:
            # Shape only — needed to confirm buffers are sized correctly.
            if isinstance(ref_tensor, torch.Tensor):
                rows, cols = ref_tensor.shape[-2], ref_tensor.shape[-1]
            else:
                rows, cols = ref_tensor.shape
        _stages["convert"] += _pc() - _t

        self._ensure_buffers(rows, cols)

        # 3) Image copy into uncached DMA buffers (only when data changed).
        _t = _pc()
        need_addr_update = False
        if need_copy:
            self.input_flt_buffer[:] = flt_np
            self.input_ref_buffer[:] = ref_np
            self._last_ref_sig = curr_ref_sig
            self._last_flt_sig = curr_flt_sig
            self._registers_configured = False
            need_addr_update = True
            self._copy_count += 1
        _stages["copy"] += _pc() - _t

        # 4) Marshal the 2x3 affine transform.
        #
        # IMPORTANT: kornia.warp_affine (used in the SW path) interprets the
        # matrix as INVERSE mapping (output -> input pixel), while Xilinx
        # xf::cv::warpTransform with TRANSFORM_TYPE=AFFINE uses FORWARD
        # mapping (input -> output pixel). Invert the 2x3 here. Closed-form
        # inverse instead of cv2.invertAffineTransform — cheaper in the hot
        # loop and bit-equivalent at FP32. The bottom row [0,0,1] was set
        # once at construction time.
        _t = _pc()
        if isinstance(transform_matrix_2x3, torch.Tensor):
            t_mat_2x3 = transform_matrix_2x3.detach().cpu().numpy()
        else:
            t_mat_2x3 = transform_matrix_2x3
        a = float(t_mat_2x3[0, 0]); b = float(t_mat_2x3[0, 1]); tx = float(t_mat_2x3[0, 2])
        c = float(t_mat_2x3[1, 0]); d = float(t_mat_2x3[1, 1]); ty = float(t_mat_2x3[1, 2])
        inv_det = 1.0 / (a * d - b * c)
        buf = self.transform_buffer
        buf[0] =  d * inv_det
        buf[1] = -b * inv_det
        buf[2] = (b * ty - d * tx) * inv_det
        buf[3] = -c * inv_det
        buf[4] =  a * inv_det
        buf[5] = (c * tx - a * ty) * inv_det
        _stages["transform"] += _pc() - _t

        # 5) Program the IP registers only if something changed.
        _t = _pc()
        if not self._registers_configured or need_addr_update:
            self.ip.write(self.OFFSET_ROWS, rows)
            self.ip.write(self.OFFSET_COLS, cols)
            self.ip.write(self.OFFSET_INPUT_IMG, self.input_flt_buffer.physical_address)
            self.ip.write(self.OFFSET_INPUT_REF, self.input_ref_buffer.physical_address)
            self.ip.write(self.OFFSET_MI_OUT, self.mi_hls_buffer.physical_address)
            self.ip.write(self.OFFSET_TRANSFORM, self.transform_buffer.physical_address)
            self._registers_configured = True
            self._regs_count += 1
        _stages["regs"] += _pc() - _t

        # 6) Kick the accelerator
        _t = _pc()
        self.ip.write(self.OFFSET_CTRL, 0x01)
        _stages["kick"] += _pc() - _t

        # 7) Poll for AP_DONE (bit 1). Timeout after 5 s. The deadline check
        # is amortized — only consult the wall clock every CHECK_EVERY reads
        # (each MMIO read is ~20-50 µs of Python overhead, so we don't want
        # to add a time.monotonic() call on top of every one).
        _t = _pc()
        ip_read = self.ip.read
        ctrl = self.OFFSET_CTRL
        polls = 0
        CHECK_EVERY = 32
        deadline = None
        while not (ip_read(ctrl) & 0x02):
            polls += 1
            if polls % CHECK_EVERY == 0:
                if deadline is None:
                    deadline = time.monotonic() + 5.0
                elif time.monotonic() > deadline:
                    try:
                        self.ip.write(self.OFFSET_CTRL, 0x00)
                    except Exception:
                        pass
                    self.enabled = False
                    self.init_error = "AP_DONE timeout (FPGA kernel stalled)"
                    raise RuntimeError(self.init_error)
        self._poll_reads += polls + 1
        _stages["poll"] += _pc() - _t

        # 8) Return the MI estimate
        _t = _pc()
        result = float(self.mi_hls_buffer[0])
        _stages["read"] += _pc() - _t
        self._call_count += 1
        return result

    def close(self):
        """Release the DMA buffers held by this accelerator."""
        for name in ("input_flt_buffer", "input_ref_buffer",
                     "transform_buffer", "mi_hls_buffer"):
            buf = getattr(self, name, None)
            if buf is not None:
                try:
                    buf.freebuffer()
                except Exception:
                    pass
