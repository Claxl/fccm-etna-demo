# -*- coding: utf-8 -*-
# /******************************************
#  * MIT License
#  *
#  * Copyright (c) 2022 Eleonora D'Arnese, Davide Conficconi, Emanuele Del
#  * Sozzo, Luigi Fusco, Donatella Sciuto, Marco Domenico Santambrogio
#  *
#  * Permission is hereby granted, free of charge, to any person obtaining a
#  * copy of this software and associated documentation files (the "Software"),
#  * to deal in the Software without restriction, including without limitation
#  * the rights to use, copy, modify, merge, publish, distribute, sublicense,
#  * and/or sell copies of the Software, and to permit persons to whom the
#  * Software is furnished to do so, subject to the following conditions:
#  *
#  * The above copyright notice and this permission notice shall be included
#  * in all copies or substantial portions of the Software.
#  ******************************************/
"""
ETNA Hyper-parameters and common numerical helpers.

Contains the default tuning constants used by the One-plus-One and Powell
optimizers and the Marsaglia-style normal variate generator reused from the
original Faber software implementation.
"""

import numpy as np


class FaberHyperParams:
    """Container for the default hyper-parameters used by ETNA optimizers."""

    # --- One-Plus-One parameters ---
    oneplusone_Maximize = False
    oneplusone_Epsilon = 1.5e-4
    oneplusone_GrowthFactor = 1.05
    oneplusone_ShrinkFactor = np.power(oneplusone_GrowthFactor, -0.25)
    oneplusone_InitialRadius = 1.01
    oneplusone_MaximumIteration = 100

    # --- Powell / Golden-Section Search parameters ---
    gss_optimaze_ths = 0.005
    powell_optimize_eps = 0.000005
    powell_rng_1 = 40.0
    powell_rng_2 = 40.0
    powell_rng_3 = 1.0
    powell_MaximumIteration = 100


class FaberOneplusoneCommonFunctions(object):
    """
    Helper class encapsulating the random-number utilities needed by the
    legacy One-plus-One optimizer. Ported verbatim from the original Faber
    software to preserve bit-exact reproducibility of published results.
    """

    def __init__(self):
        object.__init__(self)
        self.m_Gaussfaze = 1
        self.m_Gausssave = np.zeros((1, 8 * 128))
        self.dat = np.zeros((1, 6))
        self.m_GScale = 1.0 / 30000000.0

    def NormalVariateGenerator(self):
        """Return a Gaussian-distributed sample (N(0, 1))."""
        self.m_Gaussfaze = self.m_Gaussfaze - 1
        if self.m_Gaussfaze:
            return self.m_GScale * self.m_Gausssave[self.m_Gaussfaze]
        else:
            return self.FastNorm()

    def SignedShiftXOR(self, x):
        """Signed left shift + feedback XOR, mimicking the C reference impl."""
        uirs = np.uint32(x)
        c = np.int32((uirs << 1) ^ 333556017) if np.int32(x <= 0) else np.int32(uirs << 1)
        return c

    def FastNorm(self):
        """
        Marsaglia-style fast Gaussian sampler.

        This is a direct Python port of the original C implementation used in
        the first Faber release and is kept identical to reproduce published
        numerical results exactly.
        """
        m_Scale = 30000000.0
        m_Rscale = 1.0 / m_Scale
        m_Rcons = 1.0 / (2.0 * 1024.0 * 1024.0 * 1024.0)
        m_LEN = 128
        m_TLEN = 8 * m_LEN

        self.m_GScale = m_Rscale
        fake = 1.0 + 0.125 / m_TLEN
        m_Chic2 = np.sqrt(2.0 * m_TLEN - fake * fake) / fake
        m_Chic1 = fake * np.sqrt(0.5 / m_TLEN)

        m_Lseed = 12345
        m_Irs = 12345
        ts = 0.0
        p = 0

        # Build an initial pool of samples
        m_Vec1 = np.zeros(m_TLEN)
        while True:
            while True:
                m_Lseed = np.int32(69069 * np.int64(m_Lseed) + 33331)
                m_Irs = np.int64(self.SignedShiftXOR(m_Irs))
                r = np.int32(m_Irs + np.int64(m_Lseed))
                tx = m_Rcons * r
                m_Lseed = np.int32(69069 * np.int64(m_Lseed) + 33331)
                m_Irs = np.int64(self.SignedShiftXOR(m_Irs))
                r = np.int32(m_Irs + np.int64(m_Lseed))
                ty = m_Rcons * r
                tr = tx * tx + ty * ty
                if tr <= 1.0 and tr >= 0.1:
                    break
            m_Lseed = np.int32(69069 * np.int64(m_Lseed) + 33331)
            m_Irs = np.int64(self.SignedShiftXOR(m_Irs))
            r = np.int32(m_Irs + np.int64(m_Lseed))
            if r < 0:
                r = -r
            tz = m_Rcons * r
            ts += tr
            m_Vec1[p] = tz * tx
            p += 1
            m_Vec1[p] = tz * ty
            p += 1
            if p >= m_TLEN:
                break

        ts = np.sqrt(ts / m_TLEN)
        for i in range(m_TLEN):
            m_Vec1[i] /= ts

        self.m_Gausssave = m_Vec1
        self.m_Gaussfaze = m_TLEN - 1
        return self.m_GScale * self.m_Gausssave[self.m_Gaussfaze]
