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

    def __init__(self, overlay_path: str = _DEFAULT_OVERLAY):
        # Skip re-initialization (singleton semantics).
        if self._initialized:
            return

        self.enabled = False

        # Cache state: used to avoid redundant AXI-Lite register writes.
        self._last_ref_id = None
        self._last_flt_id = None
        self._registers_configured = False

        if not PYNQ_AVAILABLE:
            self._initialized = True
            return

        try:
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
                    raise RuntimeError("No suitable accelerator IP found.")

            # AXI-Lite register offsets exposed by the HLS core
            self.OFFSET_CTRL      = 0x00
            self.OFFSET_INPUT_IMG = 0x10
            self.OFFSET_INPUT_REF = 0x1c
            self.OFFSET_MI_OUT    = 0x28
            self.OFFSET_TRANSFORM = 0x34
            self.OFFSET_ROWS      = 0x40
            self.OFFSET_COLS      = 0x48

            self.current_shape = (0, 0)
            self.input_flt_buffer = None
            self.input_ref_buffer = None

            # Uncached DMA buffers for the transform matrix and MI result.
            # Uncached allocation avoids the need to flush caches manually and
            # is the simplest way to guarantee coherency with the HLS core.
            self.transform_buffer = allocate(shape=(16,), dtype=np.float32, cacheable=False)
            self.mi_hls_buffer = allocate(shape=(16,), dtype=np.float32, cacheable=False)

            self.enabled = True
            logger.info("FPGA Accelerator Ready (Optimized Uncached Mode).")

        except Exception as e:
            logger.error(f"FPGA Init Failed: {e}")
            self.enabled = False

        self._initialized = True

    def _ensure_buffers(self, rows: int, cols: int):
        """Lazily (re)allocate the input image buffers if the shape changed."""
        if (rows, cols) != self.current_shape:
            if self.input_flt_buffer:
                self.input_flt_buffer.freebuffer()
            if self.input_ref_buffer:
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
            return 0.0

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

        # 6) Busy wait for the done flag (faster than sleep for short kernels)
        while not (self.ip.read(self.OFFSET_CTRL) & 0x02):
            pass

        # 7) Return the MI estimate
        return float(self.mi_hls_buffer[0])

    def close(self):
        """Release the DMA buffers held by this accelerator."""
        try:
            if self.input_flt_buffer:
                self.input_flt_buffer.freebuffer()
            if self.input_ref_buffer:
                self.input_ref_buffer.freebuffer()
            if self.transform_buffer:
                self.transform_buffer.freebuffer()
            if self.mi_hls_buffer:
                self.mi_hls_buffer.freebuffer()
        except Exception:
            pass
