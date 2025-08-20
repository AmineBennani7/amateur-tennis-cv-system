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

# clicked_points debe existir a nivel módulo
clicked_points = []

def calibrate_homography(video_path, max_display_w=1280, max_display_h=720):
    """Show a downscaled frame for clicking 4 corners, store clicks in original coords, compute H."""
    # Lee un frame válido (intenta varios por seguridad)
    cap = cv2.VideoCapture(video_path)
    frame = None
    for _ in range(20):
        ret, temp = cap.read()
        if not ret:
            break
        frame = temp
    cap.release()
    if frame is None:
        raise RuntimeError("Failed to read a valid frame for calibration.")

    orig_h, orig_w = frame.shape[:2]

    # Escala dinámica: no ampliar, solo reducir si es muy grande
    sx = max_display_w / orig_w
    sy = max_display_h / orig_h
    SCALE = min(1.0, sx, sy)

    disp_w = int(orig_w * SCALE)
    disp_h = int(orig_h * SCALE)
    display_frame = cv2.resize(frame, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

    # Limpia clics anteriores
    global clicked_points
    clicked_points = []

    # Callback: guarda en coordenadas originales
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 4:
            xo = int(x / SCALE)
            yo = int(y / SCALE)
            clicked_points.append((xo, yo))

    cv2.namedWindow("Select 4 court corners", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Select 4 court corners", disp_w, disp_h)
    cv2.setMouseCallback("Select 4 court corners", mouse_callback)

    # Bucle de pintado
    while True:
        vis = display_frame.copy()
        for i, pt in enumerate(clicked_points):
            pt_scaled = (int(pt[0] * SCALE), int(pt[1] * SCALE))
            cv2.circle(vis, pt_scaled, 5, (0, 0, 255), -1)
            cv2.putText(vis, str(i + 1), pt_scaled, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Select 4 court corners", vis)

        if len(clicked_points) == 4:
            break
        key = cv2.waitKey(20) & 0xFF
        if key == 27:  # ESC
            cv2.destroyAllWindows()
            raise RuntimeError("Calibration cancelled by user.")

    cv2.destroyAllWindows()

    # Homografía con puntos en resolución original
    src_pts = np.array(clicked_points, dtype=np.float32)
    dst_pts = REFERENCE_POINTS.astype(np.float32)  # usa tus puntos de referencia existentes

    H, _ = cv2.findHomography(src_pts, dst_pts, method=cv2.RANSAC)
    if H is None:
        raise RuntimeError("Homography computation failed.")

    os.makedirs("calibration", exist_ok=True)
    np.save("calibration/homography_matrix.npy", H)
    print("Homography matrix calculated and saved.")
    return H