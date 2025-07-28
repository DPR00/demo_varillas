import cv2
from copy import deepcopy

class Tracker:
    def __init__(self, center_points_cur_frame, resized_frame, cam_params, debug = False):
        self.resized_frame = resized_frame
        self.cam_params = cam_params
        self.center_points_cur_frame =  center_points_cur_frame
        self.points_zone_init = [point for point in center_points_cur_frame if point.pos_x < cam_params.counter_init]
        self.points_zone_tracking = [point for point in center_points_cur_frame
                                     if cam_params.counter_end >= point.pos_x >= cam_params.counter_init]
        self.points_zone_end = [point for point in center_points_cur_frame if point.pos_x > cam_params.counter_end]
        self.debug = debug

    def update_params(self, track_id, tracking_objects, center_points_prev_frame):
        self.track_id = track_id
        self.tracking_objects = tracking_objects
        self.points_zone_init_prev = [point for point in center_points_prev_frame \
                                      if point.pos_x < self.cam_params.counter_init]
        self.points_zone_tracking_prev = [point for point in center_points_prev_frame \
                                          if self.cam_params.counter_end >= point.pos_x >= self.cam_params.counter_init]
        self.points_zone_end_prev = [point for point in center_points_prev_frame \
                                     if point.pos_x > self.cam_params.counter_end]
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
        cv2.line(self.resized_frame, (self.cam_params.counter_init, 0),
                 (self.cam_params.counter_init, self.cam_params.h), (0, 255, 0), 2)
        cv2.line(self.resized_frame, (self.cam_params.counter_end, 0),
                 (self.cam_params.counter_end, self.cam_params.h), (0, 0, 255), 2)

        if self.debug:
            print("TO: ", self.tracking_objects)

        for point in self.center_points_cur_frame:
            color = (0, 255, 0)
            if point.pos_x > self.cam_params.counter_end:
                color = (0, 0, 255)
            elif point.pos_x < self.cam_params.counter_init:
                color = (255, 0, 0)
            cv2.circle(self.resized_frame, (point.pos_x, point.pos_y), 10, color, -1)

        for object_id, point in self.tracking_objects.items():
            cv2.putText(self.resized_frame, str(object_id), (point.pos_x, point.pos_y - 7),
                        0, 1, (0, 0, 0), 2)

        text = f"Varillas: {self.track_id - 1}"

        # Get the size of the text
        (text_width, text_height), _ = cv2.getTextSize(text, self.cam_params.font,
                                                       self.cam_params.font_scale,
                                                       self.cam_params.font_thickness)

        # Position the text in the top-right corner
        text_x = self.cam_params.w - text_width - 10  # 10 pixels from the right edge
        text_y = 30  # 10 pixels from the top edge

        # Put the text on the image
        cv2.putText(self.resized_frame, text, (text_x, text_y), self.cam_params.font,
                    self.cam_params.font_scale, self.cam_params.text_color,
                    self.cam_params.font_thickness)