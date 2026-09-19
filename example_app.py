from cv_shared.video_source import load_video_source, open_video_capture
import cv_shared.append_value_to_file
from pathlib import Path
import find_n_points
import cv2

CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "camera_config/camera_config.json"
)
DEFAULT_VIDEO_SOURCE = 0
MAX_READ_FAILURES = 30
DEBUG = True
USE_CANNY_EDGE_DETECTOR = False  # True to be investigated


def main() -> None:
    prefix = "main"
    read_failures = 0
    cap = open_video_capture(load_video_source(CONFIG_PATH, DEFAULT_VIDEO_SOURCE))
    if DEBUG:
        DEBUG_DATA_FILENAME = "debug.output"
        open(DEBUG_DATA_FILENAME, "w").close()

    while True:
        display = None
        ret, frame = cap.read()

        if not ret:
            read_failures += 1
            if read_failures >= MAX_READ_FAILURES:
                print("Поток не отдаёт кадры, останавливаю захват")
                break
            # Пауза перед следующей попыткой; waitKey заодно прокачивает очередь
            # событий окна, поэтому оно не «зависает», и даёт выйти по 'q'
            # if cv2.waitKey(READ_RETRY_DELAY_MS) & 0xFF == ord('q'):
            #    break
            continue
        else:
            read_failures = 0

        cv_shared.append_value_to_file("---next frame---", DEBUG_DATA_FILENAME)
        gray = find_n_points.convert_to_gray(frame, debug=DEBUG)  # +
        bin = None
        if USE_CANNY_EDGE_DETECTOR:
            bin = find_n_points.do_canny_edge_detection_not_fully_ready(
                gray,
                threshold_1=50,
                threshold_2=100,
                reduce_noise=True,
                debug=DEBUG,
            )  #
        else:
            bin = find_n_points.binarize(
                gray,
                threshold=64,
                reduce_noise=True,
                file_path=DEBUG_DATA_FILENAME,
                debug=DEBUG,
            )  # +

        contours = find_n_points.find_holes_contours(
            bin,
            min_area_pixels=1000,
            min_solidity=0.95,
            max_solidity=0.995,  # in order to exclude real hulls in the environment
            debug=DEBUG,
            file_path=DEBUG_DATA_FILENAME,
        )  # +
        top_holes = find_n_points.find_two_top_holes(
            contours,
            # to be estimated at the closest distance for the largest deviation angle (0.71?)
            min_center_y_upper_by_lower_hole_ratio=0.71,
            file_path=DEBUG_DATA_FILENAME,
            debug=DEBUG,
        )  # +
        if (top_holes is not None) and len(top_holes) == 2:
            two_top_hulls = find_n_points.approximate_two_back_holes(top_holes)  # +
            ordered_hulls = find_n_points.order_two_hulls(two_top_hulls)  # +
            camera_matrix_coordinates_of_virtual_angles = (
                find_n_points.order_virtual_angles_coordinates_for_two_back_hulls(
                    left_hull=ordered_hulls.hole_hull_one,
                    right_hull=ordered_hulls.hole_hull_two,
                    file_path=DEBUG_DATA_FILENAME,
                    debug=DEBUG,
                )
            )  # +
            display = find_n_points.draw_final_contours_and_approximating_hulls(
                frame,
                top_holes,
                ordered_hulls,
                camera_matrix_coordinates_of_virtual_angles,
                debug=DEBUG,
            )  # +
        cv2.imshow(f"{prefix}", frame if display is None else display)
        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
