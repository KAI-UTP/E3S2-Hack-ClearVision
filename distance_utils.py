from typing import Optional


# Camera focal length in pixels.
# If distance output is too large, decrease this value.
# If distance output is too small, increase this value.
FOCAL_LENGTH_PX = 250


# Approximate real object heights in meters
REAL_OBJECT_HEIGHTS_M = {
    "person":        1.70,
    "bicycle":       1.10,
    "motorcycle":    1.20,
    "car":           1.50,
    "bus":           3.00,
    "truck":         3.00,
    "train":         3.50,
    "boat":          2.00,
    "chair":         0.90,
    "bench":         0.90,
    "couch":         0.90,
    "dining table":  0.80,
    "traffic light": 0.80,
    "stop sign":     0.75,
    "fire hydrant":  0.60,
    "dog":           0.50,
    "cat":           0.35,
    # Architectural
    "door":          2.10,
    "gate":          1.80,
    "wall":          2.00,
    "stairs":        0.80,
}


DEFAULT_OBJECT_HEIGHT_M = 1.00


def estimate_distance_m(
    label: str,
    bbox_xyxy: list[float],
    focal_length_px: float = FOCAL_LENGTH_PX,
    real_height_m: Optional[float] = None,
) -> Optional[float]:
    """
    Estimate distance from camera to detected object.

    Input:
        label: object name, e.g. "person", "car"
        bbox_xyxy: bounding box in pixel format [x1, y1, x2, y2]

    Output:
        distance in meters, e.g. 1.25
    """
    if bbox_xyxy is None or len(bbox_xyxy) != 4:
        return None

    x1, y1, x2, y2 = bbox_xyxy
    bbox_height_px = y2 - y1

    if bbox_height_px <= 0:
        return None

    object_real_height_m = real_height_m or REAL_OBJECT_HEIGHTS_M.get(
        label.lower(),
        DEFAULT_OBJECT_HEIGHT_M,
    )

    distance_m = (object_real_height_m * focal_length_px) / bbox_height_px
    return round(distance_m, 2)


def distance_to_proximity(distance_m: Optional[float]) -> str:
    """Convert metric distance to a proximity label used by the rest of the app."""
    if distance_m is None:
        return "nearby"
    if distance_m <= 1.5:
        return "close"
    if distance_m <= 4.0:
        return "nearby"
    return "far"


def distance_label(distance_m: Optional[float]) -> str:
    """Human-readable distance string, e.g. '1.2 meters'."""
    if distance_m is None:
        return ""
    if distance_m < 1.0:
        return f"{int(distance_m * 100)} centimeters"
    return f"{distance_m:.1f} meters"