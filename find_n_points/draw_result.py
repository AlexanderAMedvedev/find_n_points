import cv2
import cv_shared


def draw_result(
    input_frame: cv2.Mat,
    holes_contours: list[cv2.Mat] | None,
    hulls_approximating_holes: list[cv2.Mat] | None,
    hulls_angles_coordinates: list | None,
) -> cv2.Mat:
    left_top = (50, 100)
    center = (800, 600)
    text_color = (0, 0, 255)
    display = cv_shared.put_on_frame(
        "Look after:\n\
            1 - green filled area for holes\n\
            2 - blue lines for hulls\n\
            3 - red ordered 1-8 numbers for angles",
        text_color,
        input_frame,
        left_top,
    )

    if (
        (holes_contours is None)
        and (hulls_approximating_holes is None)
        and (hulls_angles_coordinates is None)
    ):
        display = cv_shared.put_on_frame(
            "NOTHING IS FOUND",
            text_color,
            display,
            center,
        )
        return display

    holes_contours_color = (0, 255, 0)
    holes_approximating_hulls_color = (255, 0, 0)
    holes_approximating_hulls_thickness = 2
    if holes_contours is not None:
        display = cv2.drawContours(
            input_frame,
            holes_contours,
            -1,
            holes_contours_color,
            -1,
        )
    if hulls_approximating_holes is not None:
        display = cv2.drawContours(
            display,
            hulls_approximating_holes,
            -1,
            holes_approximating_hulls_color,
            holes_approximating_hulls_thickness,
        )
    if hulls_angles_coordinates is not None:
        # assign angles of the left hole and right hole
        i = 1
        for point in hulls_angles_coordinates:
            display = cv_shared.put_on_frame(
                str(i),
                text_color,
                display,
                point,
            )
            i += 1
    return display
