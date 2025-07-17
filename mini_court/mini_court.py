import cv2
import sys
sys.path.append('../')
import constants
from utils import (
    convert_meters_to_pixel_distance,
   
)
import numpy as np


class MiniCourt():
    def __init__(self, frame):
        self.drawing_rectangle_width = 90     # width of the mini-court overlay in pixels
        self.drawing_rectangle_height = 180    # height of the mini-court overlay in pixels

        self.buffer = 30             # space between the mini-court and the video edge
        self.padding_court = 3       # padding inside the mini-court box

        self.set_canvas_background_box_position(frame)  # compute bounding box for minimap
        self.set_mini_court_position()                  # compute actual drawable court inside the box
        self.set_court_drawing_key_points()             # compute 28 keypoints for tennis court lines
        self.set_court_lines()                          # set pairs of lines to draw between keypoints

    def set_canvas_background_box_position(self, frame):
        """
        Defines the position of the outer rectangle that will contain the mini-court.
        """
        frame = frame.copy()
        self.end_x = frame.shape[1] - self.buffer  # right-most x position
        self.end_y = self.buffer + self.drawing_rectangle_height  # bottom y position of the box
        self.start_x = self.end_x - self.drawing_rectangle_width  # left-most x position
        self.start_y = self.end_y - self.drawing_rectangle_height  # top y position of the box

    def set_mini_court_position(self):
        """
        Calculates the position of the actual court drawing inside the outer box.
        """
        self.court_start_x = self.start_x + self.padding_court
        self.court_start_y = self.start_y + self.padding_court
        self.court_end_x = self.end_x - self.padding_court
        self.court_end_y = self.end_y - self.padding_court
        self.court_drawing_width = self.court_end_x - self.court_start_x  # total width of drawable court

    def convert_meters_to_pixels(self, meters):
        """
        Converts meters to pixels relative to court size.
        """
        return convert_meters_to_pixel_distance(meters,
                                                constants.DOUBLE_LINE_WIDTH,
                                                self.court_drawing_width)

    def set_court_drawing_key_points(self):
        """
        Manually sets 28 keypoints of the mini-court (x,y pairs) in pixel coordinates.
        """
        drawing_key_points = [0]*28  # flat list [x0, y0, x1, y1, ..., x13, y13]

        # Top left and top right
        drawing_key_points[0], drawing_key_points[1] = int(self.court_start_x), int(self.court_start_y)
        drawing_key_points[2], drawing_key_points[3] = int(self.court_end_x), int(self.court_start_y)

        # Bottom left and right baselines (half court)
        drawing_key_points[4] = int(self.court_start_x)
        drawing_key_points[5] = self.court_start_y + self.convert_meters_to_pixels(constants.HALF_COURT_LINE_HEIGHT * 2)
        drawing_key_points[6] = drawing_key_points[0] + self.court_drawing_width
        drawing_key_points[7] = drawing_key_points[5]

        # Doubles sidelines
        drawing_key_points[8] = drawing_key_points[0] + self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[9] = drawing_key_points[1]
        drawing_key_points[10] = drawing_key_points[4] + self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[11] = drawing_key_points[5]
        drawing_key_points[12] = drawing_key_points[2] - self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[13] = drawing_key_points[3]
        drawing_key_points[14] = drawing_key_points[6] - self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[15] = drawing_key_points[7]

        # No-man's land and service box lines
        drawing_key_points[16] = drawing_key_points[8]
        drawing_key_points[17] = drawing_key_points[9] + self.convert_meters_to_pixels(constants.NO_MANS_LAND_HEIGHT)
        drawing_key_points[18] = drawing_key_points[16] + self.convert_meters_to_pixels(constants.SINGLE_LINE_WIDTH)
        drawing_key_points[19] = drawing_key_points[17]
        drawing_key_points[20] = drawing_key_points[10]
        drawing_key_points[21] = drawing_key_points[11] - self.convert_meters_to_pixels(constants.NO_MANS_LAND_HEIGHT)
        drawing_key_points[22] = drawing_key_points[20] + self.convert_meters_to_pixels(constants.SINGLE_LINE_WIDTH)
        drawing_key_points[23] = drawing_key_points[21]

        # Center service line tops
        drawing_key_points[24] = int((drawing_key_points[16] + drawing_key_points[18]) / 2)
        drawing_key_points[25] = drawing_key_points[17]
        drawing_key_points[26] = int((drawing_key_points[20] + drawing_key_points[22]) / 2)
        drawing_key_points[27] = drawing_key_points[21]

        self.drawing_key_points = drawing_key_points  # Save to class

    def set_court_lines(self):
        """
        Sets the index pairs that define the court lines.
        """
        self.lines = [
            (0, 2),  # top base line
            (4, 5),  # left side
            (6, 7),  # right side
            (1, 3),  # bottom baseline
            (0, 1),  # top edge
            (8, 9),  # inside left doubles
            (10, 11),  # inside left
            (2, 3)   # bottom edge
        ]

    def draw_background_rectangle(self, frame):
        """
        Draws a semi-transparent white rectangle in the top-right corner of the frame.
        """
        shapes = np.zeros_like(frame, np.uint8)  # black canvas
        cv2.rectangle(
            shapes,
            (self.start_x, self.start_y),
            (self.end_x, self.end_y),
            (255, 255, 255),
            cv2.FILLED
        )
        out = frame.copy()
        alpha = 0.5
        mask = shapes.astype(bool)
        out[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]
        return out

    def draw_mini_court(self, frames):
        """
        Draws the mini court over a sequence of frames.
        """
        output_frames = []
        for frame in frames:
            frame = self.draw_background_rectangle(frame)  # translucent background
            frame = self.draw_court(frame)                # draw lines and keypoints
            output_frames.append(frame)
        return output_frames

    def draw_court(self, frame):
        """
        Draws the court keypoints and lines onto a single frame.
        """
        for i in range(0, len(self.drawing_key_points), 2):
            x = int(self.drawing_key_points[i])
            y = int(self.drawing_key_points[i + 1])
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)  # small red dot

        for line in self.lines:
            start_point = (int(self.drawing_key_points[line[0] * 2]), int(self.drawing_key_points[line[0] * 2 + 1]))
            end_point = (int(self.drawing_key_points[line[1] * 2]), int(self.drawing_key_points[line[1] * 2 + 1]))
            cv2.line(frame, start_point, end_point, (0, 0, 0), 1)  # black court lines

        # Draw net across the center of the court
        net_start_point = (self.drawing_key_points[0], int((self.drawing_key_points[1] + self.drawing_key_points[5]) / 2))
        net_end_point = (self.drawing_key_points[2], int((self.drawing_key_points[1] + self.drawing_key_points[5]) / 2))
        cv2.line(frame, net_start_point, net_end_point, (255, 0, 0), 1)  # blue net line

        return frame

    def get_start_point_of_mini_court(self):
        return (self.court_start_x, self.court_start_y)

    def get_width_of_mini_court(self):
        return self.court_drawing_width

    def get_court_drawing_keypoints(self):
        return self.drawing_key_points

    def draw_points_on_mini_court(self, frames, positions, color=(0, 255, 0)):
        """
        Draws points (e.g. ball/player positions) on the mini court in each frame.
        """
        for frame_num, frame in enumerate(frames):
            for _, position in positions[frame_num].items():
                x, y = map(int, position)
                cv2.circle(frame, (x, y), 3, color, -1)  # small green dot
        return frames
