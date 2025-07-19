import cv2
import numpy as np
import os

SCALE = 0.7
REFERENCE_POINTS = np.array([
    [0, 0],
    [10.97, 0],
    [0, 11.89],
    [10.97, 11.89]
], dtype=np.float32)
clicked_points = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 4:
        x_real = int(x / SCALE)
        y_real = int(y / SCALE)
        clicked_points.append([x_real, y_real])
        print(f"Point {len(clicked_points)} selected: ({x_real}, {y_real})")

def calibrate_homography(video_path):
    cap = cv2.VideoCapture(video_path)
    frame = None
    for _ in range(10):
        ret, temp_frame = cap.read()
        if not ret:
            break
        frame = temp_frame
    cap.release()
    if frame is None:
        raise RuntimeError("Failed to read a valid frame for calibration.")

    cv2.namedWindow("Select 4 court corners")
    cv2.setMouseCallback("Select 4 court corners", mouse_callback)

    while True:
        display_frame = frame.copy()
        resized_frame = cv2.resize(display_frame, (0, 0), fx=SCALE, fy=SCALE)
        for i, pt in enumerate(clicked_points):
            pt_scaled = (int(pt[0] * SCALE), int(pt[1] * SCALE))
            cv2.circle(resized_frame, pt_scaled, 5, (0, 0, 255), -1)
            cv2.putText(resized_frame, str(i + 1), pt_scaled, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.imshow("Select 4 court corners", resized_frame)

        if len(clicked_points) == 4:
            break
        if cv2.waitKey(1) & 0xFF == 27:
            cv2.destroyAllWindows()
            raise RuntimeError("Calibration cancelled by user.")
    
    cv2.destroyAllWindows()
    src_pts = np.array(clicked_points, dtype=np.float32)
    dst_pts = REFERENCE_POINTS
    H, _ = cv2.findHomography(src_pts, dst_pts)
    os.makedirs("calibration", exist_ok=True)
    np.save("calibration/homography_matrix.npy", H)
    print("Homography matrix calculated and saved.")
    return H
