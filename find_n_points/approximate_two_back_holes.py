from typing import NamedTuple

import cv2


class TwoBackHulls(NamedTuple):
    hole_hull_one: cv2.Mat
    hole_hull_two: cv2.Mat


def approximate_two_back_holes(
    input_holes: cv2.Mat[cv2.Mat], debug: bool = False
) -> TwoBackHulls:
    quantity_of_corners = 4

    hole_hull_one = cv2.approxPolyN(input_holes[0], quantity_of_corners)
    hole_hull_two = cv2.approxPolyN(input_holes[1], quantity_of_corners)
    return TwoBackHulls(hole_hull_one, hole_hull_two)
