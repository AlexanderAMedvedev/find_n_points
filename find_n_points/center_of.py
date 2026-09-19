from typing import Tuple
import cv2


def center_of(hole_contour: cv2.Mat) -> Tuple[float, float]:
    (cx, cy), _, _ = cv2.minAreaRect(hole_contour)
    return (cx, cy)