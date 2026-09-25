import cv_shared
from pathlib import Path
import find_n_points
import cv2

CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "camera_config/camera_config.json"
)
DEFAULT_VIDEO_SOURCE = 0
MAX_READ_FAILURES = 30
DEBUG = True
DEBUG_FILEPATH = "debug.output"
USE_CANNY_EDGE_DETECTOR = False  # True to be investigated
REDUCE_NOISE = True
BINARIZE_THRESHOLD = 32
MIN_AREA_PIXELS = 500
MIN_SOLIDITY = 0.95
MAX_SOLIDITY = 0.995
MIN_CENTER_Y_UPPER_BY_LOWER_HOLE_RATIO = 0.9


def main() -> None:
    prefix = "main"
    read_failures = 0
    cap = cv_shared.open_video_capture(
        cv_shared.load_video_source(CONFIG_PATH, DEFAULT_VIDEO_SOURCE)
    )
    if DEBUG:
        open(DEBUG_FILEPATH, "w").close()

    while True:
        display = None
        isOk, frame = cap.read()

        if not isOk:
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

        cv_shared.append_value_to_file("---next frame---", DEBUG_FILEPATH)
        camera_matrix_coordinates_of_virtual_angles, top_holes, ordered_hulls = (
            find_n_points.find_n_points_pipeline(
                frame=frame,
                debug=DEBUG,
                debug_filepath=DEBUG_FILEPATH,
                use_canny_edge_detector=USE_CANNY_EDGE_DETECTOR,
                reduce_noise=REDUCE_NOISE,
                binarize_threshold=BINARIZE_THRESHOLD,
                min_area_pixels=MIN_AREA_PIXELS,
                min_solidity=MIN_SOLIDITY,
                max_solidity=MAX_SOLIDITY,
                min_center_y_upper_by_lower_hole_ratio=MIN_CENTER_Y_UPPER_BY_LOWER_HOLE_RATIO,
            )
        )  # +
        display = find_n_points.draw_result(
            frame,
            top_holes,
            ordered_hulls,
            camera_matrix_coordinates_of_virtual_angles,
        )  # +
        cv2.imshow(f"{prefix}", display)
        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
