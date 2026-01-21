"""Vision-based Coordinate Estimator.

This module provides tools for estimating 3D relative coordinates from
2D image observations using pinhole camera model back-projection.

No simulator metadata is used - all coordinates are derived purely from
visual information (bounding boxes) and camera intrinsic parameters.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import math


@dataclass
class BoundingBox:
    """Represents a 2D bounding box in pixel coordinates.

    Attributes:
        x_min: Left edge of the bounding box (pixels)
        y_min: Top edge of the bounding box (pixels)
        x_max: Right edge of the bounding box (pixels)
        y_max: Bottom edge of the bounding box (pixels)
        label: Object class/type label
        confidence: Detection confidence score (0-1)
    """
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    label: str
    confidence: float = 1.0

    @property
    def center_x(self) -> float:
        """Get the horizontal center of the bounding box."""
        return (self.x_min + self.x_max) / 2

    @property
    def center_y(self) -> float:
        """Get the vertical center of the bounding box."""
        return (self.y_min + self.y_max) / 2

    @property
    def bottom_center_y(self) -> float:
        """Get the bottom edge Y coordinate (where object touches ground)."""
        return self.y_max

    @property
    def width(self) -> float:
        """Get the width of the bounding box."""
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        """Get the height of the bounding box."""
        return self.y_max - self.y_min


@dataclass
class RelativeCoordinate:
    """Represents a 3D position relative to the agent.

    Coordinate System (Agent-Centric):
        - x_local: Horizontal offset (positive = RIGHT, negative = LEFT)
        - z_local: Depth/distance (positive = FRONT/FORWARD)
        - y_local: Vertical offset (positive = UP)

    All values are in meters.
    """
    x_local: float  # Right (+) / Left (-)
    z_local: float  # Front (+) / Back (-)
    y_local: float = 0.0  # Up (+) / Down (-)
    object_type: str = ""
    confidence: float = 1.0

    def __str__(self) -> str:
        direction_x = "right" if self.x_local >= 0 else "left"
        return (
            f"{self.object_type}: {abs(self.x_local):.2f}m {direction_x}, "
            f"{self.z_local:.2f}m ahead"
        )


class VisionCoordinateEstimator:
    """Estimates 3D relative coordinates from 2D bounding boxes.

    Uses pinhole camera model to back-project 2D pixel coordinates
    to 3D world coordinates relative to the agent.

    This approach does NOT use simulator metadata - it relies purely
    on visual information and camera intrinsic parameters.
    """

    def __init__(self,
                 image_width: int = 800,
                 image_height: int = 600,
                 fov_degrees: float = 90.0,
                 camera_height: float = 1.57,
                 camera_tilt: float = 0.0):
        """Initialize the coordinate estimator.

        Args:
            image_width: Image width in pixels
            image_height: Image height in pixels
            fov_degrees: Horizontal field of view in degrees
            camera_height: Camera height above ground in meters
            camera_tilt: Camera tilt angle in degrees (0 = level, + = looking down)
        """
        self.image_width = image_width
        self.image_height = image_height
        self.fov_degrees = fov_degrees
        self.camera_height = camera_height
        self.camera_tilt = camera_tilt

        # Calculate focal length: f = (width / 2) / tan(fov / 2)
        self.focal_length = (image_width / 2) / math.tan(math.radians(fov_degrees / 2))

        # Image center coordinates
        self.center_x = image_width / 2
        self.center_y = image_height / 2

    def estimate_from_bbox(self, bbox: BoundingBox) -> RelativeCoordinate:
        """Estimate 3D relative coordinates from a bounding box.

        Uses the Ground Plane Assumption: the bottom of the bounding box
        is assumed to touch the ground plane, allowing depth estimation.

        Args:
            bbox: The 2D bounding box to convert

        Returns:
            RelativeCoordinate with estimated x_local and z_local
        """
        # 1. Get pixel coordinates
        cx = bbox.center_x  # Horizontal center
        cy_bottom = bbox.bottom_center_y  # Bottom edge (ground contact point)

        # 2. Calculate horizontal angle (azimuth) from center
        dx_pixel = cx - self.center_x
        angle_x = math.atan(dx_pixel / self.focal_length)

        # 3. Estimate depth (z_local) using ground plane assumption
        z_local = self._estimate_depth_from_ground_plane(cy_bottom)

        # 4. Calculate horizontal offset (x_local)
        # x_local = z * tan(angle_x)
        x_local = z_local * math.tan(angle_x)

        return RelativeCoordinate(
            x_local=round(x_local, 2),
            z_local=round(z_local, 2),
            object_type=bbox.label,
            confidence=bbox.confidence
        )

    def _estimate_depth_from_ground_plane(self, y_pixel: float) -> float:
        """Estimate depth from the Y pixel coordinate using ground plane geometry.

        When the camera looks straight ahead (tilt = 0), objects on the ground
        appear lower in the image when they are closer.

        The relationship is: depth = camera_height / tan(pitch_angle)
        where pitch_angle = atan((y_pixel - center_y) / focal_length)

        Args:
            y_pixel: The Y pixel coordinate of the ground contact point

        Returns:
            Estimated depth in meters
        """
        # Calculate vertical offset from image center
        dy_pixel = y_pixel - self.center_y

        # Calculate pitch angle (angle below/above horizon)
        pitch_angle = math.atan(dy_pixel / self.focal_length)

        # Add camera tilt if any
        total_pitch = pitch_angle + math.radians(self.camera_tilt)

        # Avoid division by zero or negative values
        if total_pitch <= 0.01:
            # Object is at or above horizon - very far away
            return 10.0  # Return a large default distance

        # Calculate depth: d = h / tan(pitch)
        depth = self.camera_height / math.tan(total_pitch)

        # Clamp to reasonable range (0.3m to 10m)
        depth = max(0.3, min(depth, 10.0))

        return depth

    def estimate_from_screen_position(self,
                                       normalized_x: float,
                                       normalized_y: float,
                                       object_type: str = "") -> RelativeCoordinate:
        """Estimate coordinates from normalized screen position.

        This is useful when VLM provides position as normalized coordinates
        rather than pixel bounding boxes.

        Args:
            normalized_x: X position (0 = left edge, 1 = right edge)
            normalized_y: Y position (0 = top edge, 1 = bottom edge)
            object_type: The object type/label

        Returns:
            RelativeCoordinate with estimated position
        """
        # Convert normalized to pixel coordinates
        pixel_x = normalized_x * self.image_width
        pixel_y = normalized_y * self.image_height

        # Create a point-like bounding box
        bbox = BoundingBox(
            x_min=pixel_x - 10,
            y_min=pixel_y - 10,
            x_max=pixel_x + 10,
            y_max=pixel_y + 10,
            label=object_type
        )

        return self.estimate_from_bbox(bbox)

    def batch_estimate(self, bboxes: List[BoundingBox]) -> List[RelativeCoordinate]:
        """Estimate coordinates for multiple bounding boxes.

        Args:
            bboxes: List of bounding boxes to process

        Returns:
            List of RelativeCoordinate estimates
        """
        return [self.estimate_from_bbox(bbox) for bbox in bboxes]

    def format_for_prompt(self, coordinates: List[RelativeCoordinate]) -> str:
        """Format coordinates as text for LLM prompt injection.

        Args:
            coordinates: List of relative coordinates to format

        Returns:
            Formatted string describing object positions
        """
        if not coordinates:
            return "No objects detected in current view."

        lines = ["## Estimated Object Positions (Vision-Based):"]
        for coord in coordinates:
            direction = "right" if coord.x_local >= 0 else "left"
            lines.append(
                f"- {coord.object_type}: approximately {abs(coord.x_local):.1f}m "
                f"to your {direction}, {coord.z_local:.1f}m ahead"
            )

        return "\n".join(lines)


class VLMBoundingBoxParser:
    """Parses bounding box outputs from various VLM formats."""

    @staticmethod
    def parse_normalized_bbox(response: str) -> List[BoundingBox]:
        """Parse bounding boxes from VLM response in normalized format.

        Expects format like:
        "Fridge: [0.1, 0.2, 0.3, 0.8]"
        "Sink: [0.5, 0.3, 0.7, 0.9]"

        Where values are [x_min, y_min, x_max, y_max] in normalized (0-1) range.

        Args:
            response: VLM response text containing bounding boxes

        Returns:
            List of BoundingBox objects
        """
        import re
        bboxes = []

        # Pattern: "Label: [x_min, y_min, x_max, y_max]"
        pattern = r'(\w+):\s*\[([0-9.]+),\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)\]'

        for match in re.finditer(pattern, response):
            label = match.group(1)
            x_min = float(match.group(2))
            y_min = float(match.group(3))
            x_max = float(match.group(4))
            y_max = float(match.group(5))

            # Convert normalized to pixel coordinates (assuming 800x600)
            bbox = BoundingBox(
                x_min=x_min * 800,
                y_min=y_min * 600,
                x_max=x_max * 800,
                y_max=y_max * 600,
                label=label
            )
            bboxes.append(bbox)

        return bboxes

    @staticmethod
    def parse_pixel_bbox(response: str, image_width: int = 800,
                         image_height: int = 600) -> List[BoundingBox]:
        """Parse bounding boxes in pixel coordinate format.

        Expects format like:
        "Fridge: [80, 120, 240, 480]"

        Where values are [x_min, y_min, x_max, y_max] in pixels.

        Args:
            response: VLM response text containing bounding boxes
            image_width: Image width for validation
            image_height: Image height for validation

        Returns:
            List of BoundingBox objects
        """
        import re
        bboxes = []

        pattern = r'(\w+):\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'

        for match in re.finditer(pattern, response):
            label = match.group(1)
            x_min = float(match.group(2))
            y_min = float(match.group(3))
            x_max = float(match.group(4))
            y_max = float(match.group(5))

            # Validate bounds
            x_min = max(0, min(x_min, image_width))
            y_min = max(0, min(y_min, image_height))
            x_max = max(0, min(x_max, image_width))
            y_max = max(0, min(y_max, image_height))

            bbox = BoundingBox(
                x_min=x_min,
                y_min=y_min,
                x_max=x_max,
                y_max=y_max,
                label=label
            )
            bboxes.append(bbox)

        return bboxes
