from pathlib import Path
import cv2
import cv_shared.append_value_to_file

from find_n_points.center_of import center_of


def find_two_top_holes(
    input_holes: cv2.Mat[cv2.Mat] | None,
    min_center_y_upper_by_lower_hole_ratio: float,
    file_path: Path,
    debug: bool = False,
) -> cv2.Mat[cv2.Mat] | None:

    prefix = "fun:find_two_top_holes"

    if input_holes is None or len(input_holes) <= 1:
        return

    two_top_holes = sorted(input_holes, key=lambda hole: center_of(hole)[1])[:2]
    # keep in mind, that the OY axis grows from up to down
    # (i.e. inversed in comparison with school's maths):
    # so the upper hole (if look at the holes) has smaller center_y value
    center_y_upper_hole = center_of(two_top_holes[0])[1]
    center_y_lower_hole = center_of(two_top_holes[1])[1]
    ratio = center_y_upper_hole / center_y_lower_hole
    if debug:
        cv_shared.append_value_to_file(
            f"{prefix}\n\
                center_y 1(upper): {center_y_upper_hole:.2f}\n\
                center_y 2(lower): {center_y_lower_hole:.2f}\n\
                upper/lower ratio: {ratio:.2f}\n\
                (min_center_y_upper_by_lower_hole_ratio: {min_center_y_upper_by_lower_hole_ratio})",
            file_path,
        )

    return two_top_holes if ratio > min_center_y_upper_by_lower_hole_ratio else None
