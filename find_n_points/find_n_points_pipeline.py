from pathlib import Path
import cv2
from find_n_points import (
    convert_to_gray,
    binarize,
    do_canny_edge_detection_not_fully_ready,
    find_holes_contours,
    find_two_top_holes,
    approximate_two_back_holes,
    order_two_hulls,
    order_virtual_angles_coordinates_for_two_back_hulls,
)


def find_n_points_pipeline(
    *,
    frame: cv2.Mat,
    debug: bool,
    debug_filepath: Path,
    use_canny_edge_detector: bool,
    reduce_noise: bool,
    binarize_threshold: bool,
    min_area_pixels: int,
    min_solidity: float,
    max_solidity: float,
    min_center_y_upper_by_lower_hole_ratio: float,
) -> tuple[list | None, list[cv2.Mat], list[cv2.Mat]] | tuple[None, None, None]:
    result_none_tuple = (None, None, None)
    gray = convert_to_gray(frame, debug)  # +
    bin = None
    if use_canny_edge_detector:
        bin = do_canny_edge_detection_not_fully_ready(
            gray,
            threshold_1=50,
            threshold_2=100,
            reduce_noise=reduce_noise,
            debug=debug,
        )  # to be investigated and tested
    else:
        bin = binarize(
            gray,
            threshold=binarize_threshold,
            reduce_noise=reduce_noise,
            file_path=debug_filepath,
            debug=debug,
        )  # +
    if bin is None:
        return result_none_tuple
    contours = find_holes_contours(
        bin,
        min_area_pixels=min_area_pixels,
        min_solidity=min_solidity,
        max_solidity=max_solidity,  # exclude real hulls in environment
        debug=debug,
        file_path=debug_filepath,
    )  # +
    if contours is None:
        return result_none_tuple

    top_holes = find_two_top_holes(
        contours,
        # to be estimated at the closest distance for the largest deviation angle
        min_center_y_upper_by_lower_hole_ratio=min_center_y_upper_by_lower_hole_ratio,
        file_path=debug_filepath,
        debug=debug,
    )  # +
    if top_holes is None:
        return result_none_tuple
    if (top_holes is not None) and len(top_holes) == 2:
        two_top_hulls = approximate_two_back_holes(top_holes)  # +
        ordered_hulls = order_two_hulls(two_top_hulls)  # +
        camera_matrix_coordinates_of_virtual_angles = (
            order_virtual_angles_coordinates_for_two_back_hulls(
                left_hull=ordered_hulls.hole_hull_one,
                right_hull=ordered_hulls.hole_hull_two,
                file_path=debug_filepath,
                debug=debug,
            )
        )  # +
        return (
            camera_matrix_coordinates_of_virtual_angles,
            top_holes,
            ordered_hulls,
        )
    return result_none_tuple
