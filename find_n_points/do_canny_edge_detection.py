import cv2


def do_canny_edge_detection_not_fully_ready(
    gray_image: cv2.Mat,
    threshold_1: float,
    threshold_2: float,
    reduce_noise: bool,
    debug: bool = False,
) -> cv2.Mat:
    prefix = "func:do_canny_edge_detection"
    if reduce_noise:
        blurred_gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    gray_image = blurred_gray_image if reduce_noise else gray_image
    image_with_edges = cv2.Canny(gray_image, threshold_1, threshold_2)
    if debug:
        cv2.imshow(f"{prefix}", image_with_edges)
    return image_with_edges
