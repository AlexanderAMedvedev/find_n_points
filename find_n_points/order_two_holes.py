import cv2
from find_n_points.approximate_two_back_holes import TwoBackHulls
from find_n_points.center_of import center_of


def order_two_hulls(input_hulls: TwoBackHulls) -> TwoBackHulls:
    # sort in the order: left hull, right hull
    from_left_to_right_ordered_hulls = sorted(
        input_hulls, key=lambda hull: center_of(hull)[0]
    )  # +

    return TwoBackHulls(
        hole_hull_one=from_left_to_right_ordered_hulls[0],
        hole_hull_two=from_left_to_right_ordered_hulls[1],
    )
