# -*- coding: utf-8 -*-
"""
ETNA — pyramidal (``ETNA_Multi``) registration engine plus the higher-level
pipeline runner that chains ETNA_Multi with feature-based detectors such as
XFeat.

Hybrid CPU/FPGA Mutual-Information registration package focused on the
pyramidal (multi-resolution) registration path. The non-pyramidal Faber
implementation lives in a separate ``faber_fpga`` git submodule
(https://github.com/necst/faber_fpga) and is consumed through
``wrappers/faber_wrapper.py``; ETNA keeps only the pyramidal scheduler, the
shared hyper-parameters, the ``wax_mi_accel`` FPGA overlay driver, and the
multi-stage pipeline executor used to combine ETNA_Multi with feature-based
front-ends.

Authors:
    Claudio Di Salvo, Emanuele Del Sozzo, Giuseppe Sorrentino,
    Eleonora D'Arnese, Paolo Panicucci, Davide Conficconi.

Public API:

- ``EtnaMultiMetric``                 — pyramidal MI/MSE/CC metric component
- ``EtnaMultiPowell``                 — pyramidal Powell optimizer
- ``EtnaMultiOnePlusOne``             — pyramidal 1+1 evolutionary optimizer
- ``AdaptiveParameters``, ``ImagePyramid``
- ``FaberFPGAAccelerator``            — singleton PYNQ overlay driver
- ``FaberHyperParams``, ``FaberOneplusoneCommonFunctions``
- ``PipelineExecutor``, ``StageCache`` — multi-stage ETNA pipeline runner
"""
from .hyperparams import FaberHyperParams, FaberOneplusoneCommonFunctions
from .fpga_accelerator import FaberFPGAAccelerator

# Pyramidal metric and optimizers (ETNA_Multi).
from .registrators_pyramidal import EtnaMultiMetric
from .optimizers_pyramidal import (
    EtnaMultiPowell,
    EtnaMultiOnePlusOne,
    AdaptiveParameters,
    ImagePyramid,
)

# Multi-stage pipeline runner (ETNA_Multi + XFeat combinations).
from .pipeline import PipelineExecutor, StageCache

__all__ = [
    # Shared hyper-params
    "FaberHyperParams",
    "FaberOneplusoneCommonFunctions",
    # FPGA driver
    "FaberFPGAAccelerator",
    # Pyramidal (ETNA_Multi) variants
    "EtnaMultiMetric",
    "EtnaMultiPowell",
    "EtnaMultiOnePlusOne",
    "AdaptiveParameters",
    "ImagePyramid",
    # Pipeline runner
    "PipelineExecutor",
    "StageCache",
]

__version__ = "1.2.0"
