import numpy as np
from main_utils.visualisation import apply_homography

def estimate_ball_speeds(ball_positions, H, fps):
    """
    Estima la velocidad de la pelota entre cada par de frames consecutivos,
    usando homografía y distancia real (en metros).

    Args:
        ball_positions: Lista de tuplas (x, y) en píxeles
        H: Matriz de homografía
        fps: Frames por segundo del vídeo

    Returns:
        Listado de velocidades estimadas [v0, v1, ..., vn] en m/s (v0 = 0)
    """
    speeds = [0.0]
    last_valid_proj = None

    for i in range(1, len(ball_positions)):
        x1, y1 = ball_positions[i - 1]
        x2, y2 = ball_positions[i]

        if x1 <= 0 or y1 <= 0 or x2 <= 0 or y2 <= 0:
            speeds.append(0.0)
            continue

        try:
            p1 = apply_homography((x1, y1), H)
            p2 = apply_homography((x2, y2), H)
        except:
            speeds.append(0.0)
            continue

        # Distancia en metros
        dist = np.linalg.norm(np.array(p2) - np.array(p1))

        # Tiempo entre frames
        delta_t = 1.0 / fps

        speed = dist / delta_t  # m/s
        speeds.append(speed)

    return speeds
