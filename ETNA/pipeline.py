# -*- coding: utf-8 -*-
"""
ETNA multi-stage pipeline runner.

ETNA bundles the pyramidal Faber (``ETNA_Multi``) registration engine together
with feature-based front-ends such as XFeat — and their combinations. The
``PipelineExecutor`` defined in this module is what actually chains two or
more stages into a single end-to-end registration: typically a coarse
feature-based alignment followed by ``ETNA_Multi`` refinement, or vice-versa.

STAR-Bench treats each registration backend as a single, atomic method and
therefore intentionally does not know about multi-stage chaining. Users who
want to run a composite ETNA pipeline import ``PipelineExecutor`` from this
module directly and pass in an instantiated detector factory.

Authors:
    Claudio Di Salvo, Emanuele Del Sozzo, Giuseppe Sorrentino,
    Eleonora D'Arnese, Paolo Panicucci, Davide Conficconi.
"""
import logging
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class StageCache:
    """Simple cache that stores results of shared first-stage detectors."""

    def __init__(self):
        self.cache = {}

    def get_key(self, stage_name: str, fixed_path: str, moving_path: str) -> str:
        fixed_abs = str(Path(fixed_path).resolve())
        moving_abs = str(Path(moving_path).resolve())
        return f"{stage_name}_{fixed_abs}_{moving_abs}"

    def get(self, key: str):
        return self.cache.get(key)

    def set(self, key: str, value: dict):
        self.cache[key] = value

    def clear(self):
        self.cache.clear()


def _compose_transforms(M_initial, M_refinement):
    """Compose two 2x3 affine matrices: ``M_final = M_refinement @ M_initial``."""
    if M_initial is None:
        return M_refinement
    if M_refinement is None:
        return M_initial

    M1_hom = np.vstack([M_initial, [0, 0, 1]])
    M2_hom = np.vstack([M_refinement, [0, 0, 1]])
    return (M2_hom @ M1_hom)[:2, :]


class PipelineExecutor:
    """Run a multi-stage ETNA registration pipeline end-to-end.

    Each stage is a detector produced by ``detector_factory.create_detector``.
    The pipeline string is a comma-separated list of stage names, e.g.
    ``"xfeat,etna_multi_powell_mi"``. The transform of each stage is composed
    with the running transform, and the warped moving image is fed to the
    next stage via a temporary file.
    """

    def __init__(self, pipeline_config: str, detector_factory, device: str,
                 cache: StageCache = None, **kwargs):
        self.pipeline_config = pipeline_config
        self.factory = detector_factory
        self.device = device
        self.cache = cache
        self.kwargs = kwargs
        self.stages = self._parse_config()

    def _parse_config(self):
        stage_names = self.pipeline_config.split(',')
        detectors = []
        for name in stage_names:
            try:
                detectors.append(
                    self.factory.create_detector(name, self.device, **self.kwargs)
                )
            except Exception as e:
                logger.error(f"Failed to create stage '{name}' for the pipeline: {e}")
                raise
        return detectors

    def run(self, fixed_path: str, moving_path: str) -> dict:
        results = {
            'stages': [],
            'failure_stage': 0,
            'pipeline': self.pipeline_config,
        }
        M_composed = None
        current_moving_path = moving_path
        temp_files = []

        for i, stage_detector in enumerate(self.stages):
            logger.info(
                f"Running stage {i + 1}/{len(self.stages)}: {stage_detector.name.upper()}"
            )
            stage_result = None

            if i == 0 and self.cache is not None:
                cache_key = self.cache.get_key(stage_detector.name, fixed_path, moving_path)
                cached = self.cache.get(cache_key)
                if cached:
                    logger.info(f"Using cached result for stage: {stage_detector.name}")
                    stage_result = cached.copy()
                else:
                    stage_result = stage_detector.register(
                        fixed_path, current_moving_path, **self.kwargs
                    )
                    if stage_result.get('transform') is not None:
                        self.cache.set(cache_key, stage_result)

            elif i > 0:
                original_moving_img = cv2.imread(moving_path, cv2.IMREAD_GRAYSCALE)
                fixed_img = cv2.imread(fixed_path, cv2.IMREAD_GRAYSCALE)
                h, w = fixed_img.shape[:2]

                if M_composed is None:
                    logger.error(
                        f"Stage {i + 1}: cannot run refinement, invalid initial transform."
                    )
                    results['failure_stage'] = i
                    break

                warped_for_refinement = cv2.warpAffine(
                    original_moving_img, M_composed, (w, h)
                )
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    cv2.imwrite(tmp.name, warped_for_refinement)
                    current_moving_path = tmp.name
                    temp_files.append(tmp.name)

                stage_result = stage_detector.register(
                    fixed_path, current_moving_path, **self.kwargs
                )

            else:
                stage_result = stage_detector.register(
                    fixed_path, current_moving_path, **self.kwargs
                )

            if stage_result is None:
                logger.critical(
                    f"Stage {i + 1} ({stage_detector.name}) returned None; aborting."
                )
                results['failure_stage'] = i + 1
                break

            results['stages'].append(stage_result)

            M_stage = stage_result.get('transform')
            if M_stage is None:
                logger.error(
                    f"Stage {i + 1} ({stage_detector.name}) produced no transform; aborting."
                )
                results['failure_stage'] = i + 1
                break

            M_composed = _compose_transforms(M_composed, M_stage)

        for f in temp_files:
            if os.path.exists(f):
                try:
                    os.unlink(f)
                except Exception as e:
                    logger.warning(f"Unable to delete temp file {f}: {e}")

        if results['stages']:
            s1 = results['stages'][0]
            results['initial_transform'] = s1.get('transform')
            results['initial_time'] = s1.get('time')
            results['keypoints_fixed'] = s1.get('keypoints_fixed')
            results['keypoints_moving'] = s1.get('keypoints_moving')
            results['NM'] = s1.get('matches_count')

        if len(results['stages']) > 1:
            s2 = results['stages'][1]
            results['refinement_matrix'] = s2.get('transform')
            results['refinement_time'] = s2.get('time')

        results['final_transform'] = (
            M_composed if results['failure_stage'] == 0 else None
        )
        return results
