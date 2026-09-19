from pathlib import Path
import cv2
import numpy as np
import cv_shared.append_value_to_file


def binarize(
    gray_image: cv2.Mat,
    reduce_noise: bool,
    threshold: int,
    file_path: Path,
    debug: bool = False,
) -> cv2.Mat:
    prefix = "fun:binarize"
    if reduce_noise:
        blurred_gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    gray_image = blurred_gray_image if reduce_noise else gray_image
    threshold_type = cv2.THRESH_BINARY
    # ordinary thresholding
    _, binarized_image = cv2.threshold(gray_image, threshold, 255, threshold_type)
    # adaptive thresholding
    # binarized_image=cv2.adaptiveThreshold(gray_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #         cv2.THRESH_BINARY,11,2) # finds both contours: inner and outer (somehow)
    # binarized_image = cv2.adaptiveThreshold(
    #    gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 2
    #) # not completely bad, but not stable if Sun is on the back fully
    if debug:
        cv2.imshow(f"{prefix} gray_image (reduce_noise={reduce_noise})", gray_image)
        _draw_histogram_for_gray_image(prefix, gray_image, threshold)
        cv_shared.append_value_to_file(
            f"{prefix} threshold: {threshold}, type: {threshold_type}", file_path
        )

    return binarized_image


def _draw_histogram_for_gray_image(
    prefix: str, input_gray_image: cv2.Mat, threshold: float
) -> None:
    histogram = cv2.calcHist([input_gray_image], [0], None, [256], (0, 256))
    max_count = float(histogram.max())

    hist_w, hist_h = 256, 400
    margin_left, margin_top, margin_right, margin_bottom = 55, 18, 15, 30
    canvas_w = margin_left + hist_w + margin_right
    canvas_h = margin_top + hist_h + margin_bottom
    histogram_image = np.full((canvas_h, canvas_w), 255, dtype=np.uint8)
    plot_bottom = margin_top + hist_h

    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = plot_bottom - int(fraction * hist_h)
        cv2.line(
            histogram_image,
            (margin_left, y),
            (margin_left + hist_w, y),
            200,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            histogram_image,
            str(int(fraction * max_count)),
            (2, y + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            0,
            1,
            cv2.LINE_AA,
        )

    for bin_value in (0, 64, 128, 192, 255):
        x = margin_left + bin_value
        cv2.line(
            histogram_image,
            (x, margin_top),
            (x, plot_bottom),
            200,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            histogram_image,
            str(bin_value),
            (max(x - 10, margin_left), canvas_h - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            0,
            1,
            cv2.LINE_AA,
        )

    for bin_index in range(256):
        bin_height = (
            int(histogram[bin_index] / max_count * hist_h) if max_count > 0 else 0
        )
        x = margin_left + bin_index
        cv2.line(
            histogram_image,
            (x, plot_bottom),
            (x, plot_bottom - bin_height),
            0,
            1,
            cv2.LINE_AA,
        )
    threshold_x = margin_left + int(threshold)
    cv2.line(
        histogram_image,
        (threshold_x, margin_top),
        (threshold_x, plot_bottom),
        128,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        histogram_image,
        str(int(threshold)),
        (min(threshold_x + 3, canvas_w - 25), margin_top + 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.32,
        128,
        1,
        cv2.LINE_AA,
    )

    cv2.imshow(f"{prefix} histogram of gray_image", histogram_image)
