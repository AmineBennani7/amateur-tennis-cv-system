import math

def measure_distance(p1, p2):
    """
    Calcula la distancia euclídea entre dos puntos (x, y).
    """
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def get_center_of_bbox(bbox):
    """
    Obtiene el centro (x, y) de una caja delimitadora [x1, y1, x2, y2].
    """
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return (cx, cy)
