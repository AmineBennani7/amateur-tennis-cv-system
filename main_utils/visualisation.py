import numpy as np
import cv2

def draw_minicourt(width=150, height=200):
    """
    Draw a realistic mini tennis court with standard line proportions.
    Returns a BGR image as a numpy array.
    """
    court = np.zeros((height, width, 3), dtype=np.uint8)
    court[:] = (0, 128, 0)  # green background

    white = (255, 255, 255)
    thickness = 2
    margin = 5

    # Dimensions of playable area
    court_width = width - 2 * margin
    court_height = height - 2 * margin

    # Outer doubles sidelines and baselines
    top_left = (margin, margin)
    bottom_right = (width - margin, height - margin)
    cv2.rectangle(court, top_left, bottom_right, white, thickness)

    # Net line (middle horizontal)
    net_y = margin + court_height // 2
    cv2.line(court, (margin, net_y), (width - margin, net_y), white, thickness)

    # Service lines (one quarter from net on each side)
    service_line_top = margin + court_height // 4
    service_line_bottom = margin + 3 * court_height // 4
    cv2.line(court, (margin, service_line_top), (width - margin, service_line_top), white, thickness)
    cv2.line(court, (margin, service_line_bottom), (width - margin, service_line_bottom), white, thickness)

    # Singles sidelines (narrower than doubles)
    singles_margin = int(court_width * 0.12)
    left_single = margin + singles_margin
    right_single = width - margin - singles_margin
    cv2.line(court, (left_single, margin), (left_single, height - margin), white, thickness)
    cv2.line(court, (right_single, margin), (right_single, height - margin), white, thickness)

    # Center service line (vertical from net to service line on both sides)
    center_x = width // 2
    cv2.line(court, (center_x, service_line_top), (center_x, net_y), white, thickness)
    cv2.line(court, (center_x, net_y), (center_x, service_line_bottom), white, thickness)

    # Center mark (short line on bottom baseline)
    center_mark_length = 8
    base_y = height - margin
    cv2.line(court, (center_x, base_y), (center_x, base_y - center_mark_length), white, thickness)

    return court


def apply_homography(point, H):
    px = np.array([point[0], point[1], 1])
    px_transformed = H @ px
    px_transformed /= px_transformed[2]
    return px_transformed[0], px_transformed[1]
