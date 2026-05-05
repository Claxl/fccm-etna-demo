# -*- coding: utf-8 -*-
"""
ETNA FPGA Accelerator.

Thin wrapper over the PYNQ overlay that exposes the ``wax_mi_accel`` IP block
used to offload the Mutual-Information + affine warp kernel.

The class is implemented as a singleton so that the overlay (a very expensive
resource) is only loaded once per process even if multiple ETNA optimizers
instantiate a metric.
"""
import logging
import os
import time
import numpy as np
import torch

logger = logging.getLogger(__name__)

try:
    from pynq import Overlay, allocate
    PYNQ_AVAILABLE = True
except ImportError:
    PYNQ_AVAILABLE = False
    logger.warning("PYNQ libraries not found. FPGA acceleration will not be available.")


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
        self._last_ref_id = None
        self._last_flt_id = None
        self._registers_configured = False

        if not PYNQ_AVAILABLE:
            self.init_error = "PYNQ not installed (pip install pynq)"
            return

        if not os.path.exists(overlay_path):
            self.init_error = f"bitstream not found at {overlay_path}"
            logger.error(f"FPGA Init Failed: {self.init_error}")
            return

        try:
            # PYNQ uses asyncio for device discovery; Streamlit's ScriptRunner
            # thread has no event loop — create one if needed.
            import asyncio
            try:
                asyncio.get_event_loop()
            except RuntimeError:
                asyncio.set_event_loop(asyncio.new_event_loop())

            logger.info(f"Loading FPGA Overlay: {overlay_path}...")
            self.overlay = Overlay(overlay_path)

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

            self.enabled = True
            self._initialized = True
            logger.info("FPGA Accelerator Ready (Optimized Uncached Mode).")

        except Exception as e:
            self.init_error = f"{type(e).__name__}: {e}"
            logger.error(f"FPGA Init Failed: {self.init_error}")
            self.enabled = False

    @property
    def status_detail(self) -> str:
        """Short human-readable status string for the UI."""
        if self.enabled:
            return f"FPGA active: overlay loaded from {self.overlay_path}"
        if self.init_error:
            return f"FPGA disabled: {self.init_error}"
        return "FPGA disabled"

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
            self._last_ref_id = None
            self._last_flt_id = None
            self._registers_configured = False

    def compute_mi(self, ref_tensor, flt_tensor, transform_matrix_2x3) -> float:
        """
        Run a single MI evaluation on the FPGA.

        This method is hot-looped by the optimizers, so it minimizes the
        amount of Python-to-AXI traffic: register writes are only issued on
        the first call or when the shape / input buffers change.
        """
        if not self.enabled:
            raise RuntimeError(self.init_error or "FPGA accelerator not enabled")

        # 1) Convert torch tensors to numpy (zero-copy when possible)
        ref_np = ref_tensor.detach().cpu().numpy() if isinstance(ref_tensor, torch.Tensor) else ref_tensor
        flt_np = flt_tensor.detach().cpu().numpy() if isinstance(flt_tensor, torch.Tensor) else flt_tensor

        rows, cols = ref_np.shape
        self._ensure_buffers(rows, cols)

        # 2) Copy image data only if the input tensors changed identity.
        curr_ref_id = id(ref_tensor)
        curr_flt_id = id(flt_tensor)
        need_addr_update = False

        if curr_ref_id != self._last_ref_id or curr_flt_id != self._last_flt_id:
            self.input_flt_buffer[:] = flt_np
            self.input_ref_buffer[:] = ref_np
            self._last_ref_id = curr_ref_id
            self._last_flt_id = curr_flt_id
            self._registers_configured = False
            need_addr_update = True

        # 3) Marshal the 2x3 affine transform into a 3x3 homogeneous matrix
        #    and store it in the uncached DMA buffer.
        if isinstance(transform_matrix_2x3, torch.Tensor):
            t_mat = transform_matrix_2x3.detach().cpu().numpy()
        else:
            t_mat = transform_matrix_2x3

        self.transform_buffer[0] = t_mat[0, 0]
        self.transform_buffer[1] = t_mat[0, 1]
        self.transform_buffer[2] = t_mat[0, 2]
        self.transform_buffer[3] = t_mat[1, 0]
        self.transform_buffer[4] = t_mat[1, 1]
        self.transform_buffer[5] = t_mat[1, 2]
        self.transform_buffer[6] = 0.0
        self.transform_buffer[7] = 0.0
        self.transform_buffer[8] = 1.0

        # 4) Program the IP registers only if something changed.
        if not self._registers_configured or need_addr_update:
            self.ip.write(self.OFFSET_ROWS, rows)
            self.ip.write(self.OFFSET_COLS, cols)
            self.ip.write(self.OFFSET_INPUT_IMG, self.input_flt_buffer.physical_address)
            self.ip.write(self.OFFSET_INPUT_REF, self.input_ref_buffer.physical_address)
            self.ip.write(self.OFFSET_MI_OUT, self.mi_hls_buffer.physical_address)
            self.ip.write(self.OFFSET_TRANSFORM, self.transform_buffer.physical_address)
            self._registers_configured = True

        # 5) Kick the accelerator
        self.ip.write(self.OFFSET_CTRL, 0x01)

        # 6) Poll for AP_DONE (bit 1).  Timeout after 5 s to avoid hanging
        # forever when the kernel stalls (e.g. DMA fault, bad address).
        _deadline = time.monotonic() + 5.0
        while not (self.ip.read(self.OFFSET_CTRL) & 0x02):
            if time.monotonic() > _deadline:
                # Force-idle the core and mark the accelerator unusable so the
                # caller falls back to software MI.
                try:
                    self.ip.write(self.OFFSET_CTRL, 0x00)
                except Exception:
                    pass
                self.enabled = False
                self.init_error = "AP_DONE timeout (FPGA kernel stalled)"
                raise RuntimeError(self.init_error)

        # 7) Return the MI estimate
        return float(self.mi_hls_buffer[0])

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
