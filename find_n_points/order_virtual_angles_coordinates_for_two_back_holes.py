from pathlib import Path
import cv2
import cv_shared.append_value_to_file
from typing import NamedTuple

from find_n_points.center_of import center_of


def order_virtual_angles_coordinates_for_two_back_hulls(
    *,
    left_hull: cv2.Mat,
    right_hull: cv2.Mat,
    file_path: Path,
    debug: bool = False,
) -> list | None:
    prefix = "func:order_virtual_angles_coordinates_for_two_back_hulls"
    ordered_angels_left_hull = _ordered_angels_for(
        left_hull, prefix, file_path, debug=debug
    )
    ordered_angels_right_hull = _ordered_angels_for(
        right_hull, prefix, file_path, debug=debug
    )
    if ordered_angels_left_hull is None or ordered_angels_right_hull is None:
        return None
    if debug:
        result_left = _angels('left', ordered_angels_left_hull)
        result_right = _angels('right', ordered_angels_right_hull)
        cv_shared.append_value_to_file(
            f"{prefix}\n {result_left}\n {result_right}", file_path
        )

    return ordered_angels_left_hull + ordered_angels_right_hull



def _ordered_angels_for(
    hull: cv2.Mat,
    prefix: str,
    file_path: Path,
    debug: bool = False,
) -> list | None:
    # sort clockwise the points of hull:
    # RT, RB, LB, LT
    hull_copy = hull.copy()
    cx, cy = center_of(hull_copy)
    if debug:
        cv_shared.append_value_to_file(f"{prefix} cx: {cx}; cy :{cy}", file_path)
    rt = None
    rb = None
    lb = None
    lt = None

    for i in range(4):
        point = hull_copy[0][i].copy()
        if debug:
            cv_shared.append_value_to_file(f"{prefix} {point}", file_path)
        x = point[0]
        y = point[1]

        if x > cx and y < cy:
            rt = point
        elif x > cx and y > cy:
            rb = point
        elif x < cx and y > cy:
            lb = point
        else:
            lt = point
    if (rt is None) or (rb is None) or (lb is None) or (lt is None):
        return None

    return [rt, rb, lb, lt]

def _angels(side: str, angels:list)->str:
    return f"""
          Angel's coordinates for the {side} hole:
          RightTop    (RT): {angels[0]}
          RightBottom (RB): {angels[1]}
          LeftBottom  (LB): {angels[2]}
          LeftTop     (LT): {angels[3]} 
        """