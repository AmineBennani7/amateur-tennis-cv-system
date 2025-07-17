def get_center_of_bbox(bbox):
    """
    the center point of a bounding box.
    """
    x1, y1, x2, y2 = bbox  # Unpack coordinates
    center_x = int((x1 + x2) / 2)  # Compute X center
    center_y = int((y1 + y2) / 2)  # Compute Y center
    return (center_x, center_y)


def measure_distance(p1, p2):
    """
     Euclidean distance between two 2D points.
    """
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) ** 0.5  # √((x2 - x1)^2 + (y2 - y1)^2)


def get_foot_position(bbox):
    """
    foot position of a player by returning the midpoint of the bottom edge of the bounding box.
    """
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), y2)


def get_closest_keypoint_index(point, keypoints, keypoint_indices):
    """
    index of the keypoint that is vertically closest to the given point.
    """
    closest_distance = float('inf')
    key_point_ind = keypoint_indices[0]

    for keypoint_index in keypoint_indices:
        keypoint = keypoints[keypoint_index * 2], keypoints[keypoint_index * 2 + 1]
        distance = abs(point[1] - keypoint[1])  # Only compare Y-coordinates

        if distance < closest_distance:
            closest_distance = distance
            key_point_ind = keypoint_index

    return key_point_ind


def get_height_of_bbox(bbox):
    """
     height of a bounding box.
    """
    return bbox[3] - bbox[1]


def measure_xy_distance(p1, p2):
    """
    absolute distance between two points in both X and Y axes.
    """
    return abs(p1[0] - p2[0]), abs(p1[1] - p2[1])


def get_center_of_bbox(bbox):
    """
    Shortcut version of center calculation.
    """
    return (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))
