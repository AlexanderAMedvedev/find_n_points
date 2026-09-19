import cv2

def draw_final_contours_and_approximating_hulls(
    input_frame: cv2.Mat,
    holes_contours: list[cv2.Mat],
    hulls_approximating_holes: list[cv2.Mat],
    hulls_angles_coordinates: list | None,
    debug: bool = False,
) -> None | cv2.Mat:
    if hulls_angles_coordinates is None:
        return None
    if debug:
        holes_contours_color = (0, 255, 0)
        holes_approximating_hulls_color = (255, 0, 0)
        holes_approximating_hulls_thickness = 2
        index_font = cv2.FONT_HERSHEY_SIMPLEX
        index_font_thickness = 1
        index_font_scale = 1
        index_color = (0, 0, 255)

        display = cv2.drawContours(
            input_frame,
            holes_contours,
            -1,
            holes_contours_color,
            -1,
        )
        display = cv2.drawContours(
            display,
            hulls_approximating_holes,
            -1,
            holes_approximating_hulls_color,
            holes_approximating_hulls_thickness,
        )
        # assign angles of the left hole and right
        i = 1
        for point in hulls_angles_coordinates:
            display = cv2.putText(
                display,
                str(i),
                point,
                index_font,
                index_font_scale,
                index_color,
                index_font_thickness,
                cv2.LINE_4,
            )
            i += 1

        return display
