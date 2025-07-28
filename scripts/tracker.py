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
        self.rods_cur_frame =  center_points_cur_frame
        self.debug = debug
        self.rods_zone_init, self.rods_zone_tracking, self.rods_zone_end = self._zone_rods(self.rods_cur_frame)

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

    def _initialize_new_tracks(self):
        """Assigns new track IDs to all rods initially in the tracking zone."""
        for rod in self.rods_zone_tracking:
            self.tracking_objects[self.track_id] = rod
            rod.track_id = self.track_id
            self.track_id += 1

    def _handle_exiting_rods(self):
        """Removes the oldest tracks if new rods appear in the end zone (FIFO logic)."""
        end_diff = len(self.rods_zone_end) - len(self.rods_zone_end_prev)
        if end_diff > 0:
            self.tracking_objects = dict(list(self.tracking_objects.items())[end_diff:])

    def _prepare_association_lists(self) -> Tuple[Dict[int, Rod], List[Rod], bool]:
        """
        Applies heuristics to decide the order and direction of matching.
        Returns copies of objects to match and a flag for the association strategy.
        """
        tracking_objects_copy = deepcopy(self.tracking_objects)
        rods_zone_tracking_copy = self.rods_zone_tracking.copy()
        use_standard_association = True

        # If rods are disappearing from the tracking zone, reverse the matching order.
        if len(self.rods_zone_tracking) < len(self.rods_zone_tracking_prev):
            tracking_objects_copy = dict(sorted(self.tracking_objects.items(), reverse=True))
            rods_zone_tracking_copy.reverse()

        # If the number of rods is stable, check for specific movement patterns.
        elif len(self.rods_zone_tracking) == len(self.rods_zone_tracking_prev):
            diffs_tracking = [a.pos_x - b.pos_x for a, b in zip(self.rods_zone_tracking, self.rods_zone_tracking_prev)]
            mean_tracking_move = sum(diffs_tracking) / len(diffs_tracking) if diffs_tracking else 0

            end_diff = len(self.rods_zone_end) - len(self.rods_zone_end_prev)
            end_is_stopped = end_diff < 0

            if end_diff == 0 and self.rods_zone_end_prev:
                diffs_end = [a.pos_x - b.pos_x for a, b in zip(self.rods_zone_end, self.rods_zone_end_prev)]
                mean_end_move = sum(diffs_end) / len(diffs_end) if diffs_end else 0
                end_is_stopped = abs(mean_end_move) < 15 # Heuristic threshold

            # Special case: If tracking rods are moving right and end zone rods have stopped,
            # use a simplified, one-to-one association strategy.
            if mean_tracking_move >= -15 and end_is_stopped: # Heuristic threshold
                tracking_objects_copy = dict(sorted(self.tracking_objects.items(), reverse=True))
                rods_zone_tracking_copy.reverse()
                use_standard_association = False

        return tracking_objects_copy, rods_zone_tracking_copy, use_standard_association

    def _associate_and_update(self,
                              objects_to_match: Dict[int, Rod],
                              rods_to_match: List[Rod],
                              use_standard_association: bool):
        """
        Matches tracked objects to current detections and updates their state.
        Handles lost tracks and creates new ones for unmatched detections.
        """
        if use_standard_association:
            unmatched_detections = rods_to_match.copy()
            lost_track_ids = []

            for object_id, rod_prev in objects_to_match.items():
                found_match = False
                for rod_curr in unmatched_detections:
                    # Motion constraint: object should not move too far backward
                    if rod_curr.pos_x - rod_prev.pos_x >= -15: # Heuristic threshold
                        self.tracking_objects[object_id] = rod_curr
                        rod_curr.track_id = object_id
                        unmatched_detections.remove(rod_curr)
                        found_match = True
                        break

                if not found_match:
                    lost_track_ids.append(object_id)

            # Clean up lost tracks
            for object_id in lost_track_ids:
                if object_id in self.tracking_objects:
                    self.tracking_objects.pop(object_id)

            # Create new tracks for remaining unmatched detections
            for rod in unmatched_detections:
                self.tracking_objects[self.track_id] = rod
                rod.track_id = self.track_id
                self.track_id += 1
        else:
            # Simplified one-to-one association for the special "stopped" case
            for object_id, _ in objects_to_match.items():
                if rods_to_match:
                    rod_curr = rods_to_match.pop(0)
                    self.tracking_objects[object_id] = rod_curr
                    rod_curr.track_id = object_id

    def update_params(self, track_id, tracking_objects, center_points_prev_frame):
        self.track_id = track_id
        self.tracking_objects = tracking_objects
        self.rods_zone_init_prev, self.rods_zone_tracking_prev, self.rods_zone_end_prev = self._zone_rods(center_points_prev_frame)

    def track(self) -> Tuple[int, Dict[int, Rod], List[Rod]]:
        """
        Performs object tracking by associating current detections with existing tracks.
        """
        # 1. If no objects are being tracked, initialize new tracks and exit.
        if not self.tracking_objects:
            self._initialize_new_tracks()
            return self.track_id, self.tracking_objects, deepcopy(self.rods_cur_frame)

        # 2. Handle objects that have exited the final zone.
        self._handle_exiting_rods()

        # 3. Prepare lists for matching based on the custom heuristics.
        # This determines the strategy for associating old tracks with new detections.
        tracking_objects_to_match, rods_to_match, use_standard_association = self._prepare_association_lists()

        # 4. Perform the association and update the state.
        self._associate_and_update(
            tracking_objects_to_match,
            rods_to_match,
            use_standard_association
        )

        return self.track_id, self.tracking_objects, deepcopy(self.rods_cur_frame)

    def plot_count(self):
        cv2.line(self.frame, (self.cp.counter_init, 0),
                 (self.cp.counter_init, self.cp.h), self.cp.green, self.cp.font_thickness)
        cv2.line(self.frame, (self.cp.counter_end, 0),
                 (self.cp.counter_end, self.cp.h), self.cp.red, self.cp.font_thickness)

        if self.debug:
            print("TO: ", self.tracking_objects)

        for point in self.rods_cur_frame:
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