# -*- coding: utf-8 -*-
"""
ETNA_Multi pyramidal optimizers.

Multi-resolution coarse-to-fine optimizers used by the ETNA_Multi metric.
Each level uses adaptive hyper-parameters (tolerance, search range, etc.)
and the transformation estimated at the coarser level is used as the initial
guess for the next finer level.

These classes are intentionally branded under the ``EtnaMulti*`` namespace
so that the pyramidal engine stays virtually separate from the non-pyramidal
Faber implementation shipped as the ``faber_fpga`` submodule.
"""
import time
import logging
from abc import ABCMeta, abstractmethod

import numpy as np
import torch
import kornia
import cv2

from .hyperparams import FaberHyperParams

logger = logging.getLogger(__name__)

# Force CPU as default device (the module targets embedded FPGA boards such
# as the Xilinx Kria KR260 where no GPU is available).
DEFAULT_DEVICE = 'cpu'


class AdaptiveParameters:
    """
    Level-aware hyper-parameter adapter.

    At the coarsest level we use a larger search range and looser tolerance;
    at the finest level we use the nominal values defined in
    ``FaberHyperParams``. The multipliers below have been tuned empirically.
    """

    @staticmethod
    def get_iterations(level: int, max_level: int, optimizer_type: str = 'powell') -> int:
        if optimizer_type == 'powell':
            base_iterations = getattr(FaberHyperParams, 'powell_MaximumIteration', 100)
        else:
            base_iterations = FaberHyperParams.oneplusone_MaximumIteration

        if level == max_level - 1:
            factor = 1.0
        else:
            normalized_level = 1.0 - (level / (max_level - 1))
            factor = 1.0 + 0.5 * normalized_level

        return max(int(base_iterations * factor), 20)

    @staticmethod
    def get_tolerance(level: int, max_level: int, optimizer_type: str = 'powell') -> float:
        base_tol = FaberHyperParams.powell_optimize_eps if optimizer_type == 'powell' \
            else FaberHyperParams.oneplusone_Epsilon

        if level == max_level - 1:
            multiplier = 1.0
        else:
            normalized_level = 1.0 - (level / (max_level - 1))
            multiplier = 1.0 - 0.7 * normalized_level

        return base_tol * multiplier

    @staticmethod
    def get_search_range(level: int, max_level: int, param_index: int) -> float:
        base_ranges = [
            FaberHyperParams.powell_rng_1,
            FaberHyperParams.powell_rng_2,
            FaberHyperParams.powell_rng_3,
        ]

        if level == max_level - 1:
            multiplier = 1.0
        else:
            normalized_level = 1.0 - (level / (max_level - 1))
            multiplier = 1.0 - 0.6 * normalized_level

        return base_ranges[param_index] * multiplier

    @staticmethod
    def get_initial_radius(level: int, max_level: int) -> float:
        base_radius = FaberHyperParams.oneplusone_InitialRadius
        if level == max_level - 1:
            multiplier = 1.0
        else:
            normalized_level = 1.0 - (level / (max_level - 1))
            multiplier = 1.0 - 0.7 * normalized_level
        return base_radius * multiplier

    @staticmethod
    def get_gss_threshold(level: int, max_level: int) -> float:
        base_threshold = FaberHyperParams.gss_optimaze_ths
        if level == max_level - 1:
            multiplier = 1.0
        else:
            normalized_level = 1.0 - (level / (max_level - 1))
            multiplier = 1.0 - 0.8 * normalized_level
        return base_threshold * multiplier


class ImagePyramid:
    """
    Gaussian image pyramid built with torch primitives.

    Each level is the previous level low-pass filtered with a 5x5 Gaussian
    kernel and downsampled by ``scale_factor``.
    """

    def __init__(self, image: torch.Tensor, levels: int = 4, scale_factor: float = 0.5, device=DEFAULT_DEVICE):
        self.levels = levels
        self.scale_factor = scale_factor
        self.device = device

        self.image_float = image.float() if image.dtype == torch.uint8 else image
        self.pyramid = self._build_pyramid(self.image_float)

    def _build_pyramid(self, image):
        pyramid = [image.byte()]
        current = image
        if current.dim() == 2:
            current = current.unsqueeze(0).unsqueeze(0)
        elif current.dim() == 3:
            current = current.unsqueeze(0)

        # Build a normalized 5x5 Gaussian kernel (sigma = 1)
        kernel_size = 5
        sigma = 1.0
        coords = torch.arange(kernel_size).float() - (kernel_size - 1) / 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        kernel_2d = g.unsqueeze(1) @ g.unsqueeze(0)
        kernel = kernel_2d.unsqueeze(0).unsqueeze(0).to(self.device)

        for i in range(1, self.levels):
            if current.shape[-1] < 2 or current.shape[-2] < 2:
                pyramid.append(pyramid[-1])
                continue

            blurred = torch.nn.functional.conv2d(current, kernel, padding=kernel_size // 2)
            downsampled = torch.nn.functional.interpolate(
                blurred, scale_factor=self.scale_factor, mode='bicubic', align_corners=False
            )
            pyramid.append(downsampled.squeeze().byte())
            current = downsampled

        return pyramid

    def get_level(self, level: int) -> torch.Tensor:
        if level >= len(self.pyramid):
            return self.pyramid[-1]
        return self.pyramid[level]

    def get_ordered_levels(self):
        """Return the level indices in coarse-to-fine order."""
        return list(reversed(range(self.levels)))


class EtnaMultiSwOptimizers(object, metaclass=ABCMeta):
    """Abstract base class for the ETNA_Multi pyramidal optimizers."""

    def __init__(self):
        self.debugger = None

    def set_debug_mode(self, debug_output_dir=None):
        pass

    @abstractmethod
    def compute(self, ref_tensor, flt_tensor, metric_component, image_dimension, use_pyramid=True):
        pass

    @abstractmethod
    def register_images(self, Ref_uint8, Flt_uint8, metric_component):
        pass

    def _normalize_to_uint8(self, tensor: torch.Tensor, device) -> torch.Tensor:
        if tensor.dtype == torch.uint8:
            return tensor.to(device)
        tensor_float = tensor.float().to(device)
        img_min, img_max = tensor_float.min(), tensor_float.max()
        if img_max > img_min:
            normalized = ((tensor_float - img_min) / (img_max - img_min) * 255.0).round()
        else:
            normalized = torch.zeros_like(tensor_float)
        return normalized.byte()

    def readprep_torch_dicom_refflt_pair(self, i, j, device):
        try:
            import pydicom
        except ImportError:
            pydicom = None

        if pydicom is not None and i.lower().endswith('.dcm'):
            ref = pydicom.dcmread(i)
            Ref_img = torch.tensor(ref.pixel_array.astype(np.int16), dtype=torch.int16, device=device)
            Ref_img[Ref_img == -2000] = 1
        else:
            ref = cv2.imread(i, 0)
            Ref_img = torch.tensor(ref.astype(np.int16), dtype=torch.int16, device=device)

        if pydicom is not None and j.lower().endswith('.dcm'):
            flt = pydicom.dcmread(j)
            Flt_img = torch.tensor(flt.pixel_array.astype(np.int16), dtype=torch.int16, device=device)
        else:
            flt = cv2.imread(j, 0)
            Flt_img = torch.tensor(flt.astype(np.int16), dtype=torch.int16, device=device)

        return self._normalize_to_uint8(Ref_img, device), self._normalize_to_uint8(Flt_img, device)

    def scale_transformation_matrix(self, H, scale_factor, direction='up'):
        """Scale only the translation component of a 2x3 affine matrix."""
        H_scaled = H.copy()
        scale = 1.0 / scale_factor if direction == 'up' else scale_factor
        H_scaled[0, 2] *= scale
        H_scaled[1, 2] *= scale
        return H_scaled

    def compute_final_metric(self, ref, flt, H, metric_component):
        """Return the metric value achieved by a given 2x3 transform."""
        ref_ravel = ref.ravel().double() if ref.dim() == 2 else ref.double()
        eref = metric_component.precompute_metric(ref_ravel)
        H_tensor = torch.from_numpy(H).float().to(metric_component.device)
        return metric_component.compute_metric(ref, flt, H_tensor, eref)

    def save_data(self, out_stack, name, res_path):
        pass


class EtnaMultiOnePlusOne(EtnaMultiSwOptimizers):
    """Pyramidal One-Plus-One optimizer."""

    def __init__(self):
        super().__init__()

    def compute_from_files(self, CT, PET, name, curr_res, t_id, patient_id, metric_component, image_dimension, use_pyramid=True):
        results = []
        for i, j in zip(CT, PET):
            Ref_uint8, Flt_uint8 = self.readprep_torch_dicom_refflt_pair(i, j, metric_component.device)
            H_final = self.compute(Ref_uint8, Flt_uint8, metric_component, image_dimension, use_pyramid)
            results.append(H_final)
        return results[0] if len(results) == 1 else results

    def compute(self, ref_tensor: torch.Tensor, flt_tensor: torch.Tensor, metric_component, image_dimension, use_pyramid=True, num_levels: int = 4):
        device = metric_component.device
        Ref_uint8 = self._normalize_to_uint8(ref_tensor, device)
        Flt_uint8 = self._normalize_to_uint8(flt_tensor, device)

        if not use_pyramid or num_levels <= 1:
            logger.info("[Pyramid] Single-level fallback")
            start_single = time.time()
            _, H = self.register_images(Ref_uint8, Flt_uint8, metric_component)
            logger.info(f"[Pyramid] Single-level time: {time.time() - start_single:.4f}s")
            return H.detach().cpu().numpy() if torch.is_tensor(H) else H

        pyramid_start = time.time()
        flt_pyramid = ImagePyramid(Flt_uint8, num_levels, 0.5, device)
        ref_pyramid = ImagePyramid(Ref_uint8, num_levels, 0.5, device)
        logger.info(f"[Pyramid] Construction time: {time.time() - pyramid_start:.4f}s")

        level_transforms = []

        for level in reversed(range(num_levels)):
            level_start = time.time()
            ref_level = ref_pyramid.get_level(level)
            flt_level = flt_pyramid.get_level(level)

            # Use the previous (coarser) transform as the starting point.
            H_init = None
            if level < num_levels - 1 and len(level_transforms) > 0:
                prev_H = level_transforms[-1]
                H_init = self.scale_transformation_matrix(prev_H, 0.5, direction='up')

            _, H_level = self.register_images_adaptive(
                ref_level, flt_level, metric_component, H_init, level,
                max_level=num_levels,
            )
            level_transforms.append(H_level.cpu().numpy())

            logger.info(f"[Pyramid] Level {level} (size {tuple(ref_level.shape)}) time: {time.time() - level_start:.4f}s")

        return level_transforms[-1]

    def register_images_adaptive(self, Ref_uint8, Flt_uint8, metric_component, H_init=None, level=0, max_level: int = 4):
        parent = torch.empty((2, 3), device=metric_component.device)
        metric_component.estimate_initial(Ref_uint8, Flt_uint8, parent)

        if H_init is not None:
            parent = torch.from_numpy(H_init).float().to(metric_component.device)

        # Precompute on flattened reference, but feed the 2-D view to the
        # metric so that the FPGA path can skip the reshape.
        Ref_uint8_ravel = Ref_uint8.ravel().double()
        eref = metric_component.precompute_metric(Ref_uint8_ravel)

        max_iter = AdaptiveParameters.get_iterations(level, max_level, 'oneplusone')
        epsilon = AdaptiveParameters.get_tolerance(level, max_level, 'oneplusone')

        optimal_params = self.OnePlusOne_adaptive(
            Ref_uint8, Flt_uint8, metric_component, eref, parent,
            max_iter, epsilon, level,
        )

        params_trans = metric_component.to_matrix_blocked(optimal_params)
        flt_transform = metric_component.transform(Flt_uint8, params_trans)
        return flt_transform, params_trans

    def register_images(self, Ref_uint8, Flt_uint8, metric_component):
        return self.register_images_adaptive(Ref_uint8, Flt_uint8, metric_component)

    def OnePlusOne_adaptive(self, Ref_uint8_2D, Flt_uint8, metric_component, eref, parent,
                            max_iterations, epsilon, level):
        """One-plus-one with early-stop patience and level-aware parameters."""
        parent_cpu = parent.cpu()
        m_Maximize = FaberHyperParams.oneplusone_Maximize
        m_GrowthFactor = FaberHyperParams.oneplusone_GrowthFactor
        m_ShrinkFactor = FaberHyperParams.oneplusone_ShrinkFactor

        initial_radius = AdaptiveParameters.get_initial_radius(level, 3)
        spaceDimension = 3
        A = torch.eye(spaceDimension, device=metric_component.device) * initial_radius

        # Recover θ from the 2x3 rigid matrix (``[[cosθ, sinθ, tx],
        # [-sinθ, cosθ, ty]]`` — ETNA/kornia inverse-mapping convention) so
        # the optimizer walks in signed angle space rather than in cos(θ).
        theta_seed = torch.atan2(parent_cpu[0][1], parent_cpu[0][0])
        parentPosition = torch.tensor(
            [parent_cpu[0][2], parent_cpu[1][2], theta_seed],
            device=metric_component.device,
        )

        # Keep the seed matrix fed to compute_metric consistent with
        # parentPosition (estimate_initial may have filled parent with
        # cos/sin that don't exactly match the ``to_matrix_blocked`` layout
        # after the atan2 round-trip — re-synthesize it).
        parent = metric_component.to_matrix_blocked(parentPosition.cpu()).to(
            metric_component.device
        )
        pvalue = metric_component.compute_metric(Ref_uint8_2D, Flt_uint8, parent, eref)

        patience = max(30, max_iterations // 3)
        no_improvement_count = 0
        best_value = pvalue
        best_position = parentPosition.clone()

        for i in range(max_iterations):
            f_norm = torch.randn(spaceDimension, device=metric_component.device)

            delta = A.matmul(f_norm)
            child = parentPosition + delta
            childPosition = metric_component.to_matrix_blocked(child)

            cvalue = metric_component.compute_metric(Ref_uint8_2D, Flt_uint8, childPosition, eref)

            adjust = m_ShrinkFactor

            if m_Maximize:
                if cvalue > pvalue:
                    pvalue, adjust, parentPosition, child = cvalue, m_GrowthFactor, child, parentPosition
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
            else:
                if cvalue < pvalue:
                    pvalue, adjust, parentPosition, child = cvalue, m_GrowthFactor, child, parentPosition
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

            # Track the best solution seen so far
            if (m_Maximize and pvalue > best_value) or (not m_Maximize and pvalue < best_value):
                best_value = pvalue
                best_position = parentPosition.clone()

            # Patience-based early stop
            if no_improvement_count > patience:
                break

            # Convergence on the search covariance
            m_FrobeniusNorm = torch.norm(A, 'fro')
            if m_FrobeniusNorm <= epsilon:
                break

            alpha = (adjust - 1.0) / torch.dot(f_norm, f_norm)
            A += alpha * torch.outer(delta, f_norm)

        return best_position.cpu()


class EtnaMultiPowell(EtnaMultiSwOptimizers):
    """Pyramidal Powell optimizer."""

    def __init__(self):
        super().__init__()

    def compute_from_files(self, CT, PET, name, curr_res, t_id, patient_id, metric_component, image_dimension, use_pyramid=True):
        results = []
        for i, j in zip(CT, PET):
            Ref_uint8, Flt_uint8 = self.readprep_torch_dicom_refflt_pair(i, j, metric_component.device)
            H_final = self.compute(Ref_uint8, Flt_uint8, metric_component, image_dimension, use_pyramid)
            results.append(H_final)
        return results[0] if len(results) == 1 else results

    def compute(self, ref_tensor: torch.Tensor, flt_tensor: torch.Tensor, metric_component, image_dimension, use_pyramid=True, num_levels: int = 4):
        device = metric_component.device
        Ref_uint8 = self._normalize_to_uint8(ref_tensor, device)
        Flt_uint8 = self._normalize_to_uint8(flt_tensor, device)

        if not use_pyramid or num_levels <= 1:
            logger.info("[Pyramid] Single-level fallback")
            start_single = time.time()
            _, H = self.register_images(Ref_uint8, Flt_uint8, metric_component)
            logger.info(f"[Pyramid] Single-level time: {time.time() - start_single:.4f}s")
            return H.detach().cpu().numpy() if torch.is_tensor(H) else H

        pyramid_start = time.time()
        flt_pyramid = ImagePyramid(Flt_uint8, num_levels, 0.5, device)
        ref_pyramid = ImagePyramid(Ref_uint8, num_levels, 0.5, device)
        logger.info(f"[Pyramid] Construction time: {time.time() - pyramid_start:.4f}s")

        ordered_levels = flt_pyramid.get_ordered_levels()
        level_transforms = {}
        prev_level = None

        for idx, level in enumerate(ordered_levels):
            level_start = time.time()
            ref_level = ref_pyramid.get_level(level)
            flt_level = flt_pyramid.get_level(level)

            # Seed initial guess: moments at the coarsest level, upscaled
            # previous transform at the finer levels.
            H_init = None
            if level == num_levels - 1:
                params_moments = torch.empty((2, 3), device=metric_component.device)
                metric_component.estimate_initial(ref_level, flt_level, params_moments)
                H_init = params_moments.cpu().numpy()
            elif idx > 0 and prev_level is not None:
                prev_H = level_transforms[prev_level]
                H_init = self.scale_transformation_matrix(prev_H, 0.5, direction='up')

            _, H_level = self.register_images_adaptive(
                ref_level, flt_level, metric_component, H_init, level,
                max_level=num_levels,
            )
            level_transforms[level] = H_level.cpu().numpy()
            prev_level = level

            logger.info(f"[Pyramid] Level {level} (size {tuple(ref_level.shape)}) time: {time.time() - level_start:.4f}s")

        return level_transforms[0]

    def register_images_adaptive(self, Ref_uint8, Flt_uint8, metric_component, H_init=None, level=0, max_iterations_override=None, max_level: int = 4):
        # Seed the 3-parameter search vector [tx, ty, θ] from the prior.
        # With the ETNA convention ``M = [[cosθ, sinθ], [-sinθ, cosθ]]``,
        # the rotation angle is recovered via atan2(M[0][1], M[0][0]).
        # Previously the code read only M[0][0] as the 3rd parameter, which
        # dropped the sign of sin and collapsed rotation to a single
        # direction; atan2 restores the signed angle.
        if H_init is not None:
            tx_seed = float(H_init[0][2])
            ty_seed = float(H_init[1][2])
            theta_seed = float(np.arctan2(float(H_init[0][1]), float(H_init[0][0])))
        else:
            tx_seed = 0.0
            ty_seed = 0.0
            theta_seed = 0.0

        base_ranges = [
            AdaptiveParameters.get_search_range(level, max_level, 0),
            AdaptiveParameters.get_search_range(level, max_level, 1),
            AdaptiveParameters.get_search_range(level, max_level, 2),
        ]

        rng = torch.tensor(base_ranges)
        pa = torch.tensor([tx_seed, ty_seed, theta_seed])

        Ref_uint8_ravel = Ref_uint8.ravel().double()
        eref = metric_component.precompute_metric(Ref_uint8_ravel)

        optimal_params = self.optimize_powell_adaptive(
            rng, pa, Ref_uint8, Flt_uint8, metric_component, eref, level,
            max_iterations_override=max_iterations_override,
            max_level=max_level,
        )

        params_trans = metric_component.to_matrix_blocked(optimal_params)
        flt_transform = metric_component.transform(Flt_uint8, params_trans)

        return flt_transform, params_trans

    def optimize_powell_adaptive(self, rng, par_lin, ref_sup_2D, flt_sup,
                                 metric_component, eref, level, max_iterations_override=None, max_level: int = 4):
        """Powell iterations with per-level tolerance and randomized parameter order."""
        converged = False
        eps = AdaptiveParameters.get_tolerance(level, max_level, 'powell')
        max_iterations = max_iterations_override or AdaptiveParameters.get_iterations(level, max_level, 'powell')

        last_mut = 100000.0
        it = 0
        best_params = par_lin.clone()
        best_metric = last_mut

        rand_gen = np.random.RandomState(420)

        while not converged and it < max_iterations:
            converged = True
            it += 1

            # At the coarse level we keep the axis order fixed so that the
            # initial sweep is deterministic; finer levels shuffle.
            param_order = [0, 1, 2]
            if level > 0:
                rand_gen.shuffle(param_order)

            for param_idx in param_order:
                cur_par = par_lin[param_idx]
                cur_rng = rng[param_idx]

                param_opt, cur_mi = self.optimize_goldsearch_adaptive(
                    cur_par, cur_rng, ref_sup_2D, flt_sup,
                    par_lin, param_idx, metric_component, eref, level,
                    max_level=max_level,
                )

                par_lin[param_idx] = cur_par
                improvement = last_mut - cur_mi
                if improvement > eps:
                    par_lin[param_idx] = param_opt
                    last_mut = cur_mi
                    converged = False
                    if cur_mi < best_metric:
                        best_metric = cur_mi
                        best_params = par_lin.clone()
                else:
                    par_lin[param_idx] = cur_par

        return best_params

    def optimize_goldsearch_adaptive(self, par, rng, ref_sup_2D, flt_sup,
                                     linear_par, i, metric_component, eref, level, max_level: int = 4):
        """Adaptive golden-section search with per-level tuning."""
        base_threshold = AdaptiveParameters.get_gss_threshold(level, max_level)
        threshold = base_threshold * 5.0

        if level == 0:
            max_it = 15
        elif level == 1:
            max_it = 25
        else:
            max_it = 40

        ratio_1 = 0.382
        ratio_2 = 0.618

        range_multiplier = 1.0
        if level == 0:
            range_multiplier = 0.3
        elif level == 1:
            range_multiplier = 0.5

        start = par - ratio_1 * rng * range_multiplier
        end = par + ratio_2 * rng * range_multiplier

        if abs(end - start) < threshold:
            return par, float('inf')

        c = end - (end - start) / 1.618
        d = start + (end - start) / 1.618

        def evaluate_at(point):
            linear_par[i] = point
            matrix = metric_component.to_matrix_blocked(linear_par)
            return metric_component.compute_metric(ref_sup_2D, flt_sup, matrix, eref)

        mi_c = evaluate_at(c)
        mi_d = evaluate_at(d)

        best_param = c if mi_c < mi_d else d
        best_metric = min(mi_c, mi_d)

        no_improvement = 0
        last_best = best_metric
        it = 0

        while abs(c - d) > threshold and it < max_it:
            if mi_c < mi_d:
                end, d, mi_d = d, c, mi_c
                c = end - (end - start) / 1.618
                mi_c = evaluate_at(c)
            else:
                start, c, mi_c = c, d, mi_d
                d = start + (end - start) / 1.618
                mi_d = evaluate_at(d)

            current_best = min(mi_c, mi_d)
            current_best_param = c if mi_c < mi_d else d

            if current_best < best_metric:
                best_metric = current_best
                best_param = current_best_param
                no_improvement = 0
            else:
                no_improvement += 1

            # Patience-based early stop
            if no_improvement >= 5:
                break
            # Auto-relax threshold if the best stopped moving
            if abs(last_best - best_metric) < threshold * 0.1:
                threshold *= 1.5

            last_best = best_metric
            it += 1

        linear_par[i] = best_param
        return best_param, -best_metric

    def register_images(self, Ref_uint8, Flt_uint8, metric_component):
        return self.register_images_adaptive(Ref_uint8, Flt_uint8, metric_component)
