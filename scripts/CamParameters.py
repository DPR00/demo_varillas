import cv2

class CameraParameters:
    def __init__(self, width, height, x = 2440, y=480, w=600, h=720):
        # Get original video dimensions
        self.width = width
        self.height = height
        self.x = x  # x-coordinate of the top-left corner of the rectangle
        self.y = y  # y-coordinate of the top-left corner of the rectangle
        self.w = w  # width of the rectangle
        self.h = h # height of the rectangle
        # For counter init and end lines
        self.counter_init = h
        self.counter_end = 0
        self.counter_line = h//2
        # For plotting in the camera frame
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.8
        self.font_thickness = 2
        self.blue = (255, 0, 0)
        self.green = (0, 255, 0)
        self.red = (0, 0, 255)
        self.black = (0, 0, 0)
        self.white = (255, 255, 255)
        self.rod_radius = 10

    def update_limits(self, counter_init, counter_end, counter_line):
        self.counter_init = counter_init
        self.counter_end = counter_end
        self.counter_line = counter_line

