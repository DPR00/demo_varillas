from .datatypes import Rod
from copy import deepcopy
from typing import List, Dict, Tuple
import cv2
import numpy as np

class Tracker:
    def __init__(self,
                 center_points_cur_frame: List[Rod],
                 frame: np.ndarray,
                 cam_params,
                 debug: bool = False):
        self.frame = frame
        self.cp = cam_params
        self.center_points_cur_frame =  center_points_cur_frame
        self.debug = debug
        self.points_zone_init, self.points_zone_tracking, self.points_zone_end = self._zone_rods(self.center_points_cur_frame)

    def _zone_rods(self, rods: List[Rod]) -> Tuple[List[Rod], List[Rod], List[Rod]]:
        """
        Sorts a list of points into three spatial zones based on their x-coordinate.

        Returns:
            A tuple containing three lists of points for the init, tracking, and end zones.
        """
        zone_init = [rod for rod in rods if rod.pos_x < self.cp.counter_init]
        zone_tracking = [rod for rod in rods if self.cp.counter_init <= rod.pos_x <= self.cp.counter_end]
        zone_end = [rod for rod in rods if rod.pos_x > self.cp.counter_end]

        return zone_init, zone_tracking, zone_end

    def update_params(self, track_id, tracking_objects, center_points_prev_frame):
        self.track_id = track_id
        self.tracking_objects = tracking_objects
        self.points_zone_init_prev, self.points_zone_tracking_prev, self.points_zone_end_prev = self._zone_rods(center_points_prev_frame)

    def track(self):
        center_points_prev_frame_copy = deepcopy(self.center_points_cur_frame)

        if len(self.tracking_objects) == 0:
            for point in self.points_zone_tracking:
                self.tracking_objects[self.track_id] = point
                point.track_id = self.track_id
                self.track_id += 1
        else:
            association = True
            init_diff = len(self.points_zone_init) - len(self.points_zone_init_prev)
            end_diff = len(self.points_zone_end) - len(self.points_zone_end_prev)
            if end_diff > 0:
                self.tracking_objects = dict(list(self.tracking_objects.items())[end_diff:])

            tracking_objects_copy = deepcopy(self.tracking_objects)
            points_zone_tracking_copy = self.points_zone_tracking.copy()

            if len(self.points_zone_tracking) > len(self.points_zone_tracking_prev):
                pass
            elif len(self.points_zone_tracking) < len(self.points_zone_tracking_prev):
                tracking_objects_copy = dict(sorted(self.tracking_objects.items(), reverse=True))
                points_zone_tracking_copy = points_zone_tracking_copy[::-1]
            else:
                diffs_tracking = [a.pos_x - b.pos_x for a, b in zip(self.points_zone_tracking, self.points_zone_tracking_prev)]
                mean_tracking_movement = sum(diffs_tracking)/len(diffs_tracking) if diffs_tracking else 0
                # print(mean_tracking_movement, "---", diffs_tracking)
                mean_tracking_pos_x_prev = sum(r.pos_x for r in self.points_zone_tracking_prev) / len(self.points_zone_tracking_prev) if self.points_zone_tracking_prev else 0
                mean_tracking_pos_x_cur = sum(r.pos_x for r in self.points_zone_tracking) / len(self.points_zone_tracking) if self.points_zone_tracking else 0
                moving_tracking_to_the_left = mean_tracking_pos_x_cur < mean_tracking_pos_x_prev
                mean_init_pos_x_prev = sum(r.pos_x for r in self.points_zone_init_prev) / len(self.points_zone_init_prev) if self.points_zone_init_prev else 0
                mean_init_pos_x_cur = sum(r.pos_x for r in self.points_zone_init) / len(self.points_zone_init) if self.points_zone_init else 0
                moving_init_to_the_left = mean_init_pos_x_cur < mean_init_pos_x_prev
                mean_end_pos_x_prev = sum(r.pos_x for r in self.points_zone_end_prev) / len(self.points_zone_end_prev) if self.points_zone_end_prev else 0
                mean_end_pos_x_cur = sum(r.pos_x for r in self.points_zone_end) / len(self.points_zone_end) if self.points_zone_end else 0
                moving_end_to_the_left = mean_end_pos_x_cur > mean_end_pos_x_prev
                end_is_stopped = end_diff < 0
                if end_diff == 0:
                    diffs_end = [a.pos_x - b.pos_x for a, b in zip(self.points_zone_end, self.points_zone_end_prev)]
                    mean_end_movement = sum(diffs_end)/len(diffs_end) if diffs_end else 0
                    end_is_stopped = abs(mean_end_movement) < 15
                    # print(end_is_stopped, "---", mean_end_movement, "---", diffs_end)
                if mean_tracking_movement >= -15 and end_is_stopped:
                    tracking_objects_copy = dict(sorted(self.tracking_objects.items(), reverse=True))
                    points_zone_tracking_copy = points_zone_tracking_copy[::-1]
                    association = False

            if association:
                for object_id, pt_prev in tracking_objects_copy.items():
                    object_exists = False
                    # if self.debug:
                    #     print(f"ID: {object_id} -> ", end="")
                    for pt_curr in points_zone_tracking_copy:
                        # print(pt2[0] - pt[0] >= -15, pt2[0] - pt[0])
                        if pt_curr.pos_x - pt_prev.pos_x >= -15:
                            distance = pt_prev.pos_x - pt_curr.pos_x

                            # if self.debug:
                            #     print(distance,", ", end="")

                            self.tracking_objects[object_id] = pt_curr
                            pt_curr.track_id = object_id

                            object_exists = True

                            # if self.debug:
                            #     print("Removed: ", end="")

                            if pt_curr in points_zone_tracking_copy:
                                # if self.debug:
                                #     print(pt, end="")
                                points_zone_tracking_copy.remove(pt_curr)
                            break

                    # Remove IDs lost
                    if not object_exists:
                        self.tracking_objects.pop(object_id)

                # Add new IDs found
                for pt in points_zone_tracking_copy:
                    self.tracking_objects[self.track_id] = pt
                    pt.track_id = self.track_id
                    self.track_id += 1
            else:
                for object_id, pt_prev in tracking_objects_copy.items():
                    for pt_curr in points_zone_tracking_copy:
                        self.tracking_objects[object_id] = pt_curr
                        pt_curr.track_id = object_id
                        if pt_curr in points_zone_tracking_copy:
                            points_zone_tracking_copy.remove(pt_curr)
                        break

        return self.track_id, self.tracking_objects, center_points_prev_frame_copy

    def plot_count(self):
        cv2.line(self.frame, (self.cp.counter_init, 0),
                 (self.cp.counter_init, self.cp.h), self.cp.green, self.cp.font_thickness)
        cv2.line(self.frame, (self.cp.counter_end, 0),
                 (self.cp.counter_end, self.cp.h), self.cp.red, self.cp.font_thickness)

        if self.debug:
            print("TO: ", self.tracking_objects)

        for point in self.center_points_cur_frame:
            color = self.cp.green
            if point.pos_x > self.cp.counter_end:
                color = self.cp.red
            elif point.pos_x < self.cp.counter_init:
                color = self.cp.blue
            cv2.circle(self.frame, (point.pos_x, point.pos_y), self.cp.rod_radius, color, -1)

        for object_id, point in self.tracking_objects.items():
            text_pos = (point.pos_x, point.pos_y - 7)
            cv2.putText(self.frame, str(object_id),text_pos, 0, 1,  self.cp.black, self.cp.font_thickness)

        text = f"Varillas: {self.track_id - 1}"

        # Get the size of the text
        (text_width, text_height), _ = cv2.getTextSize(text, self.cp.font,
                                                       self.cp.font_scale,
                                                       self.cp.font_thickness)

        # Position the text in the top-right corner
        text_x = self.cp.w - text_width - 10  # 10 pixels from the right edge
        text_y = 30  # 10 pixels from the top edge

        # Put the text on the image
        cv2.putText(self.frame, text, (text_x, text_y), self.cp.font,
                    self.cp.font_scale, self.cp.green,
                    self.cp.font_thickness)