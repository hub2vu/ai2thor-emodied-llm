"""Vision-based coordinate estimation module.

This module provides tools for estimating 3D relative coordinates
from 2D image observations without using simulator metadata.
"""

from src.vision.coordinate_estimator import (
    VisionCoordinateEstimator,
    VLMBoundingBoxParser,
    BoundingBox,
    RelativeCoordinate
)

__all__ = [
    "VisionCoordinateEstimator",
    "VLMBoundingBoxParser",
    "BoundingBox",
    "RelativeCoordinate",
]
