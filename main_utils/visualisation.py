import numpy as np
import cv2

def draw_minicourt(width=150, height=200):
    """
    Draw a mini tennis court with correct markings.
    Returns a BGR image numpy array.
    """
    court = np.zeros((height, width, 3), dtype=np.uint8)
    court[:] = (0, 128, 0)  # dark green background

    white = (255, 255, 255)
    thickness = 2

    # Outer boundary
    cv2.rectangle(court, (5, 5), (width - 5, height - 5), white, thickness)

    # Net line (horizontal middle)
    cv2.line(court, (5, height // 2), (width - 5, height // 2), white, thickness)

    # Service lines (top and bottom quarter lines)
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

def apply_homography(point, H):
    px = np.array([point[0], point[1], 1])
    px_transformed = H @ px
    px_transformed /= px_transformed[2]
    return px_transformed[0], px_transformed[1]
