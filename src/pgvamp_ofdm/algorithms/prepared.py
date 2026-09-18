"""Explicit inference-only preparation, owned by one unchanged detector and system."""

from typing import Any

import torch

from ..utils.validation import validate_system
from .base import DetectionResult
from .mmse import MMSEDetector
from .pg_vamp.majorizer import majorizer
from .pg_vamp.model import PGVAMPDetector
from .vamp import VAMPDetector

Detector = MMSEDetector | VAMPDetector | PGVAMPDetector


def settings(model: Detector) -> tuple[Any, ...]:
    if isinstance(model, PGVAMPDetector):
        return (
            model.depth,
            model.rho_hi_db,
            model.rho_lo_db,
            model.min_gap_db,
            model.temperature_db,
            model.jitter,
        )
    return (model.iterations,) if isinstance(model, VAMPDetector) else ()


class PreparedDetector:
    """Frozen private copies; reject changed H/noise/weights/settings, never cache y.

    Both construction and detection require eval and inference_mode. Validation
    belongs to the measured online cost. PG factors still depend on gamma2 and
    are recomputed in every layer; only masks and majorizer terms are reused.
    """

    def __init__(self, model: Detector, H: torch.Tensor, sigma2: torch.Tensor) -> None:
        self._guard(model)
        validate_system(H, H.new_zeros(H.shape[:2]), sigma2)
        self._model = model
        self._H, self._sigma2 = H.clone(), sigma2.clone()
        self._parameters = [p.clone() for p in model.parameters()]
        self._settings = settings(model)
        self._factor = None
        self._svd = None
        self._terms = None
        if isinstance(model, MMSEDetector):
            self._factor = model._factor(self._H, self._sigma2)
        elif isinstance(model, VAMPDetector):
            self._svd = tuple(torch.linalg.svd(self._H, full_matrices=False))
        else:
            rho, _ = model.thresholds()
            masks = [model._mask(self._H, r) for r in rho]
            self._terms = [(m, majorizer(self._H, m)) for m in masks]

    @staticmethod
    def _guard(model: Detector) -> None:
        if model.training or not torch.is_inference_mode_enabled():
            raise ValueError("prepared detection requires eval() and inference_mode()")

    def detect(self, H: torch.Tensor, y: torch.Tensor, sigma2: torch.Tensor) -> DetectionResult:
        self._guard(self._model)
        for actual, saved in ((H, self._H), (sigma2, self._sigma2)):
            if (
                actual.shape != saved.shape
                or actual.dtype != saved.dtype
                or actual.device != saved.device
                or not torch.equal(actual, saved)
            ):
                raise ValueError("prepared state invalidated by changed H/noise/shape/dtype/device")
        if settings(self._model) != self._settings or any(
            p.device != s.device or p.dtype != s.dtype or not torch.equal(p, s)
            for p, s in zip(self._model.parameters(), self._parameters, strict=True)
        ):
            raise ValueError("prepared state invalidated by changed model")
        if isinstance(self._model, MMSEDetector):
            return self._model._detect(H, y, sigma2, self._factor)
        if isinstance(self._model, VAMPDetector):
            return self._model._detect(H, y, sigma2, svd=self._svd)
        return self._model(H, y, sigma2, _prepared=self._terms, _summaries=False)


def timed_detect(
    model: Detector, H: torch.Tensor, y: torch.Tensor, sigma2: torch.Tensor
) -> DetectionResult:
    """Cold path includes all work except optional PG logging summaries."""
    if isinstance(model, PGVAMPDetector):
        return model(H, y, sigma2, _summaries=False)
    return model.detect(H, y, sigma2)
