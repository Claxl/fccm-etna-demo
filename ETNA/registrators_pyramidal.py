# -*- coding: utf-8 -*-
"""
ETNA_Multi pyramidal metric component.

Multi-resolution similarity metric (MI / MSE / CC / Parzen) used by the
``EtnaMulti*`` pyramidal optimizers. Kept as an independent module so the
pyramidal code path can evolve separately from the non-pyramidal Faber
implementation shipped as the ``faber_fpga`` submodule.
"""
import logging
import math

import numpy as np
import torch
import kornia
from scipy import signal

logger = logging.getLogger(__name__)

try:
    from .fpga_accelerator import FaberFPGAAccelerator
except ImportError:
    FaberFPGAAccelerator = None


class EtnaMultiMetric(object):
    """Pyramidal ETNA_Multi metric component (MI / MSE / CC / Parzen)."""

    def __init__(self, ref_size=512, metric="mi", transform="wax",
                 exponential=True, interpolation="nearest", use_fpga=False):
        object.__init__(self)
        self.ref_entropy = 0
        self.ref_size = ref_size
        self.metric = metric
        self.similarity_function = None
        self.transform_component = transform
        self.exponential = exponential
        self.interpolation = interpolation

        # --- FPGA setup ---
        self.use_fpga = use_fpga
        self.fpga_accel = None
        self.device = "cpu"

        if self.use_fpga:
            if FaberFPGAAccelerator:
                try:
                    self.fpga_accel = FaberFPGAAccelerator()
                    if not self.fpga_accel.enabled:
                        logger.warning("FPGA requested but failed to initialize. Falling back to software.")
                        self.use_fpga = False
                    else:
                        logger.info("FPGA acceleration ENABLED for Mutual Information (Pyramidal).")
                except Exception as e:
                    logger.warning(f"Error initializing FPGA: {e}")
                    self.use_fpga = False
            else:
                logger.warning("ETNA FPGA module not found. FPGA disabled.")
                self.use_fpga = False

        self.compute_metric = None
        self.precompute_metric = None
        self.ref_vals = torch.ones(ref_size * ref_size, dtype=torch.int, device=self.device)
        self.move_data = None
        self.hist_dim = 256
        self.map_similarity_function()
        self.move_data = self.no_transfer

    def map_similarity_function(self):
        if self.metric == "mi":
            self.compute_metric = self.compute_mi_exponential if self.exponential else self.compute_mi
            self.precompute_metric = self.precompute_mutual_information
        elif self.metric == "prz":
            self.compute_metric = self.compute_parzen_mi_exponential if self.exponential else self.parzen_mi
            self.precompute_metric = self.precompute_parzen
        elif self.metric == "cc":
            self.compute_metric = self.compute_cc
            self.precompute_metric = self.precompute_cross_correlation
        elif self.metric == "mse":
            self.compute_metric = self.compute_mse
            self.precompute_metric = self.precompute_mean_squared_error
        else:
            self.compute_metric = self.compute_mi
            self.precompute_metric = self.precompute_mutual_information

    def batch_transform(self, images, pars):
        return kornia.geometry.warp_affine(
            images, pars, mode=self.interpolation,
            dsize=(images.shape[2], images.shape[3])
        )

    def no_transfer(self, input_data):
        return input_data

    def transform(self, image, par):
        tmp_img = image.reshape((1, 1, *image.shape)).float()
        t_par = torch.unsqueeze(par, dim=0)
        return kornia.geometry.warp_affine(
            tmp_img, t_par, mode=self.interpolation,
            dsize=(tmp_img.shape[2], tmp_img.shape[3])
        )

    def my_squared_hist2d_t(self, sample, bins, smin, smax):
        D, N = sample.shape
        edges = torch.linspace(smin, smax, bins + 1, device=self.device)
        nbin = edges.shape[0] + 1

        Ncount = D * [None]
        for i in range(D):
            Ncount[i] = torch.searchsorted(edges, sample[i, :], right=True)
        for i in range(D):
            on_edge = (sample[i, :] == edges[-1])
            Ncount[i][on_edge] -= 1

        xy = Ncount[0] * nbin + Ncount[1]
        hist = torch.bincount(xy, None, minlength=nbin * nbin)
        hist = hist.reshape((nbin, nbin)).float()
        return hist[1:-1, 1:-1]

    def precompute_mutual_information(self, Ref_uint8_ravel):
        href = torch.histc(Ref_uint8_ravel, bins=256)
        href /= Ref_uint8_ravel.numel()
        href = href[href > 1e-15]
        eref = (torch.sum(href * torch.log2(href))) * -1
        return eref

    def mutual_information(self, Ref_uint8_ravel, Flt_uint8_ravel, eref):
        idx_joint = torch.stack((Ref_uint8_ravel, Flt_uint8_ravel))
        j_h_init = self.my_squared_hist2d_t(idx_joint, self.hist_dim, 0, 255) / Ref_uint8_ravel.numel()

        j_h = j_h_init[j_h_init > 1e-15]
        entropy = (torch.sum(j_h * torch.log2(j_h))) * -1

        hflt = torch.sum(j_h_init, axis=0)
        hflt = hflt[hflt > 1e-15]
        eflt = (torch.sum(hflt * torch.log2(hflt))) * -1

        return eref + eflt - entropy

    def precompute_cross_correlation(self, Ref_uint8_ravel):
        return torch.sum(Ref_uint8_ravel * Ref_uint8_ravel)

    def cross_correlation(self, Ref_uint8_ravel, Flt_uint8_ravel, cc_ref):
        cc_ref_flt = torch.sum(Ref_uint8_ravel * Flt_uint8_ravel)
        cc_flt = torch.sum(Flt_uint8_ravel * Flt_uint8_ravel)
        return -cc_ref_flt / torch.sqrt(cc_ref * cc_flt)

    def precompute_mean_squared_error(self, Ref_uint8_ravel):
        pass

    def mean_squared_error(self, Ref_uint8_ravel, Flt_uint8_ravel, mse_ref):
        return torch.sum((Ref_uint8_ravel - Flt_uint8_ravel) ** 2)

    def compute_mi(self, ref_img, flt_img, t_mat, eref):
        """FPGA-aware MI with software fallback. See standard variant for details."""
        if self.use_fpga:
            try:
                mi_val = self.fpga_accel.compute_mi(ref_img, flt_img, t_mat)
                return torch.tensor(-mi_val, device=self.device)
            except Exception as e:
                logger.warning(f"FPGA Error: {e}. Fallback to SW.")
                flt_warped = self.transform(flt_img, t_mat)
                return -(self.mutual_information(ref_img.ravel(), flt_warped.ravel(), eref).cpu())
        else:
            flt_warped = self.transform(flt_img, t_mat)
            mi = self.mutual_information(ref_img.ravel(), flt_warped.ravel(), eref)
            return -(mi.cpu())

    def compute_mi_exponential(self, ref_img, flt_img, t_mat, eref):
        mi = self.compute_mi(ref_img, flt_img, t_mat, eref)
        return torch.exp(mi).cpu()

    def compute_cc(self, ref_img, flt_img, t_mat, cc_ref):
        flt_warped = self.transform(flt_img, t_mat)
        cc = self.cross_correlation(ref_img.ravel(), flt_warped.ravel(), cc_ref)
        return cc.cpu()

    def compute_cc_exponential(self, ref_img, flt_img, t_mat, cc_ref):
        cc = self.compute_cc(ref_img, flt_img, t_mat, cc_ref)
        return torch.exp(-cc).cpu()

    def compute_mse(self, ref_img, flt_img, t_mat, mse_ref):
        flt_warped = self.transform(flt_img, t_mat)
        mse = self.mean_squared_error(ref_img.ravel(), flt_warped.ravel(), mse_ref)
        return mse.cpu()

    def compute_mi_couple(self, ref_img, flt_imgs, t_mats, eref):
        flt_warped = self.batch_transform(flt_imgs, t_mats)
        mi_a = self.mutual_information(ref_img.ravel(), flt_warped[0].ravel(), eref)
        mi_b = self.mutual_information(ref_img.ravel(), flt_warped[1].ravel(), eref)
        return torch.exp(-mi_a).cpu(), torch.exp(-mi_b).cpu()

    def compute_cc_couple(self, ref_img, flt_imgs, t_mats, cc_ref):
        flt_warped = self.batch_transform(flt_imgs, t_mats)
        cc_a = self.cross_correlation(ref_img.ravel(), flt_warped[0].ravel(), cc_ref)
        cc_b = self.cross_correlation(ref_img.ravel(), flt_warped[1].ravel(), cc_ref)
        return cc_a.cpu(), cc_b.cpu()

    def compute_mse_couple(self, ref_img, flt_imgs, t_mats, mse_ref):
        flt_warped = self.batch_transform(flt_imgs, t_mats)
        mse_a = self.mean_squared_error(ref_img.ravel(), flt_warped[0].ravel(), mse_ref)
        mse_b = self.mean_squared_error(ref_img.ravel(), flt_warped[1].ravel(), mse_ref)
        return mse_a.cpu(), mse_b.cpu()

    def compute_moments(self, img):
        h, w = img.shape
        y = torch.arange(h, device=self.device)
        x = torch.arange(w, device=self.device)
        x_grid = x.reshape(1, w).expand(h, w)
        y_grid = y.reshape(h, 1).expand(h, w)
        moments = torch.empty(6, device=self.device)
        moments[0] = torch.sum(img)
        moments[1] = torch.sum(img * x_grid)
        moments[2] = torch.sum(img * (x_grid ** 2))
        moments[3] = torch.sum(img * y_grid)
        moments[4] = torch.sum(img * (y_grid ** 2))
        moments[5] = torch.sum(img * x_grid * y_grid)
        return moments

    def to_matrix_blocked(self, vector_params):
        mat_params = torch.empty((2, 3))
        mat_params[0][2] = vector_params[0]
        mat_params[1][2] = vector_params[1]
        if vector_params[2] > 1 or vector_params[2] < -1:
            mat_params[0][0] = 1
            mat_params[1][1] = 1
            mat_params[0][1] = 0
            mat_params[1][0] = 0
        else:
            mat_params[0][0] = vector_params[2]
            mat_params[1][1] = vector_params[2]
            mat_params[0][1] = torch.sqrt(1 - (vector_params[2] ** 2))
            mat_params[1][0] = -mat_params[0][1]
        return mat_params

    def estimate_initial(self, Ref_uint8, Flt_uint8, params):
        ref_mom = self.compute_moments(Ref_uint8)
        flt_mom = self.compute_moments(Flt_uint8)

        flt_avg_10 = flt_mom[1] / flt_mom[0]
        flt_avg_01 = flt_mom[3] / flt_mom[0]
        flt_mu_20 = (flt_mom[2] / flt_mom[0] * 1.0) - (flt_avg_10 * flt_avg_10)
        flt_mu_02 = (flt_mom[4] / flt_mom[0] * 1.0) - (flt_avg_01 * flt_avg_01)
        flt_mu_11 = (flt_mom[5] / flt_mom[0] * 1.0) - (flt_avg_01 * flt_avg_10)

        ref_avg_10 = ref_mom[1] / ref_mom[0]
        ref_avg_01 = ref_mom[3] / ref_mom[0]
        ref_mu_20 = (ref_mom[2] / ref_mom[0] * 1.0) - (ref_avg_10 * ref_avg_10)
        ref_mu_02 = (ref_mom[4] / ref_mom[0] * 1.0) - (ref_avg_01 * ref_avg_01)
        ref_mu_11 = (ref_mom[5] / ref_mom[0] * 1.0) - (ref_avg_01 * ref_avg_10)

        params[0][2] = ref_mom[1] / ref_mom[0] - flt_mom[1] / flt_mom[0]
        params[1][2] = ref_mom[3] / ref_mom[0] - flt_mom[3] / flt_mom[0]

        rho_flt = 0.5 * torch.atan((2.0 * flt_mu_11) / (flt_mu_20 - flt_mu_02))
        rho_ref = 0.5 * torch.atan((2.0 * ref_mu_11) / (ref_mu_20 - ref_mu_02))
        delta_rho = rho_ref - rho_flt

        roundness = (flt_mom[2] / flt_mom[0]) / (flt_mom[4] / flt_mom[0])
        if torch.abs(roundness - 1.0) >= 0.3:
            params[0][0] = torch.cos(delta_rho)
            params[0][1] = -torch.sin(delta_rho)
            params[1][0] = torch.sin(delta_rho)
            params[1][1] = torch.cos(delta_rho)
        else:
            params[0][0] = 1.0
            params[0][1] = 0.0
            params[1][0] = 0.0
            params[1][1] = 1.0
        return params

    def precompute_parzen(self, Ref_uint8_ravel):
        pass

    def parzen_mi(self, fixed, moving, bin=256, padding=False):
        n_bins = bin
        omega = np.array([[1 / 6, 2 / 3, 1 / 6]])
        filter = np.dot(omega.transpose(), omega)
        pad_size = omega.size // 2 if padding else 0

        epsilon = np.finfo(np.float64).tiny
        count_matrix = np.zeros((n_bins, n_bins))
        fixed_clipped = np.clip(fixed, 0, n_bins - 1)
        moving_clipped = np.clip(moving, 0, n_bins - 1)
        for f, m in zip(fixed_clipped.flatten().astype(int), moving_clipped.flatten().astype(int)):
            count_matrix[m, f] += 1

        if pad_size > 0:
            count_matrix = np.pad(count_matrix, pad_size)

        prob_matrix = signal.correlate2d(count_matrix, filter, "same")
        prob_matrix /= fixed_clipped.size
        prob_k = np.sum(prob_matrix, axis=0)
        prob_j = np.sum(prob_matrix, axis=1)

        logs = np.zeros((n_bins + pad_size * 2, n_bins + pad_size * 2))
        for j in range(n_bins + pad_size * 2):
            for k in range(n_bins + pad_size * 2):
                denom = prob_j[j] * prob_k[k] or epsilon
                num = prob_matrix[j, k] or epsilon
                logs[j, k] = math.log(num / denom)

        res = 0
        for j in range(n_bins + pad_size * 2):
            for k in range(n_bins + pad_size * 2):
                res += prob_matrix[j, k] * logs[j, k]
        return -res

    def compute_parzen_mi_exponential(self, ref, flt):
        mi = self.parzen_mi(ref, flt)
        return np.exp(-mi)
