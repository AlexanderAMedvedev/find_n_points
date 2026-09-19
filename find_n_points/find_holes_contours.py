import cv2
import cv_shared.append_value_to_file
from pathlib import Path


def find_holes_contours(
    input_binarized_image: cv2.Mat,
    min_area_pixels: int,
    min_solidity: float,
    file_path: Path,
    max_solidity: float = 1.0,
    debug: bool = False,
) -> list[cv2.Mat] | None:
    contours, hierarchy = cv2.findContours(
        input_binarized_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    ) # MAY BE REPLACED BY connectedComponentsWithStats()
    prefix = "fun:find_holes_contours"
    hole_contour_color = (0, 0, 255)  # BGR: контуры отверстий — красные
    contour_thickness = 1
    # RETR_CCOMP - retrieve connected components
    # (all holes, are on the same level)
    # CHAIN_APPROX_SIMPLE - compress horizontal , vertical
    # and diagonal segments, leaving only their ending points
    if hierarchy is None:
        return None

    holes = []
    for contour, (*_, parent) in zip(contours, hierarchy[0]):
        if parent == -1:
            continue
        if cv2.contourArea(contour) < min_area_pixels:
            continue
        if (_solidityOf(contour) < min_solidity) or (
            _solidityOf(contour) >= max_solidity
        ):
            if debug:
                cv_shared.append_value_to_file(f"{prefix} contour's\n \
                parent: {parent}\n \
                solidity: {_solidityOf(contour)} ", file_path)
            continue
        holes.append(contour)
        if debug:
            cv_shared.append_value_to_file(f"{prefix} contour's\n \
                parent: {parent}\n \
                solidity: {_solidityOf(contour)} ", file_path)

    if debug:
        display_color = cv2.cvtColor(input_binarized_image, cv2.COLOR_GRAY2BGR)
        display_color = cv2.drawContours(
            display_color,
            contours,
            -1,  # -1 - draw all input contours
            hole_contour_color,
            contour_thickness,
        )
        cv2.imshow(f"{prefix} bare cv2.findContours(...) result", display_color)
        #
        display_color = cv2.cvtColor(input_binarized_image, cv2.COLOR_GRAY2BGR)
        display_color = cv2.drawContours(
            display_color,
            holes,
            -1,  # -1 - draw all input contours
            hole_contour_color,
            -1,
        )
        cv2.imshow(
            f"{prefix} bare cv2.findContours(...) result AFTER conditions",
            display_color,
        )
    if debug:
        cv_shared.append_value_to_file("-local end-", file_path)
    return holes


def _solidityOf(contour: cv2.Mat) -> float:
    """
    0<=return_value<=1
    """
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    # hull - оболочка
    # convex - выпуклый
    if hull_area == 0:
        return 0.0

    return cv2.contourArea(contour) / hull_area
