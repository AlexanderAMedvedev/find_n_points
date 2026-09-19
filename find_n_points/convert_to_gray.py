import cv2


def convert_to_gray(input_frame: cv2.Mat, debug: bool = False) -> cv2.Mat:
    prefix = "fun:convert_to_gray"
    gray_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2GRAY)
    if debug:
        cv2.imshow(f"{prefix}", gray_frame)
    return gray_frame
