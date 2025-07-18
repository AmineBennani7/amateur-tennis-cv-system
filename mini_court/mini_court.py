import cv2
import numpy as np

def draw_minicourt(width=150, height=200):
    """
    Draw a tennis mini court image with proper lines and markings.
    Returns a BGR image (numpy array) of size (height, width).
    """
    court = np.zeros((height, width, 3), dtype=np.uint8)
    court[:] = (0, 128, 0)  # dark green background

    white = (255, 255, 255)
    thickness = 2

    # Outer boundary
    cv2.rectangle(court, (5, 5), (width - 5, height - 5), white, thickness)

    # Net line (horizontal middle)
    cv2.line(court, (5, height // 2), (width - 5, height // 2), white, thickness)

    # Service line (horizontal line at 1/4 height from top and bottom)
    cv2.line(court, (5, height // 4), (width - 5, height // 4), white, thickness)
    cv2.line(court, (5, height * 3 // 4), (width - 5, height * 3 // 4), white, thickness)

    # Singles sidelines
    left_sideline = 5 + width // 8
    right_sideline = width - 5 - width // 8
    cv2.line(court, (left_sideline, 5), (left_sideline, height - 5), white, thickness)
    cv2.line(court, (right_sideline, 5), (right_sideline, height - 5), white, thickness)

    # Center service line (vertical in upper half)
    cv2.line(court, (width // 2, 5), (width // 2, height // 2), white, thickness)

    # Center mark on baseline (small vertical line)
    center_x = width // 2
    baseline_y_top = height - 5
    baseline_y_bottom = baseline_y_top - 10
    cv2.line(court, (center_x, baseline_y_top), (center_x, baseline_y_bottom), white, thickness)

    return court
