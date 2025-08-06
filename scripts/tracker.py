from .datatypes import Rod
from copy import deepcopy
from typing import List, Dict, Tuple, Set
import cv2
import numpy as np

class Tracker:
    def __init__(self,
                 center_points_cur_frame: List[Rod],
                 frame: np.ndarray,
                 cam_params,
                 direction: int,
                 debug: bool = False):
        self.rods_cur_frame =  center_points_cur_frame
        self.frame = frame
        self.cp = cam_params
        self.direction = direction
        self.debug = debug
        self.rods_zone_init, self.rods_zone_tracking, self.rods_zone_end = self._zone_rods(self.rods_cur_frame)
        self.rod_count = 0
        self.counted_track_ids: Set[int] = set()

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
        if end_diff == 1:
            self.tracking_objects = dict(list(self.tracking_objects.items())[end_diff:])

    def _prepare_association_lists(self) -> Tuple[Dict[int, Rod], List[Rod], bool]:
        """
        Applies heuristics to decide the order and direction of matching.
        Returns copies of objects to match and a flag for the association strategy.
        Parameters:
            self.tracking_objects: Objects being tracked from the previous frame
        """
        use_standard_association = True

        # --- Detect rods leaving the init zone ---
        exiting_init_zone_count = 0
        if self.rods_zone_init_prev: # Check if previous frame data exists
             exiting_init_zone_count = len(self.rods_zone_init_prev) - len(self.rods_zone_init)
        # ---

        tracking_objects_copy = deepcopy(self.tracking_objects)
        rods_zone_tracking_copy = self.rods_zone_tracking.copy()
        tracking_diff = len(self.rods_zone_tracking) - len(self.rods_zone_tracking_prev)
        end_diff = len(self.rods_zone_end) - len(self.rods_zone_end_prev)

        if len(self.rods_zone_tracking) > len(self.rods_zone_tracking_prev):
            if tracking_diff > 0 and (tracking_diff + end_diff) == 0:
                self._log("TRYING TO SOLVE EDGE CASE III.", 100, 20*4)

                for i in range(tracking_diff):
                    tmp_diff_track = self.cp.counter_end - rods_zone_tracking_copy[i].pos_x
                    tmp_diff_end = self.rods_zone_end[len(self.rods_zone_end) - (i+1)].pos_x - self.cp.counter_end if len(self.rods_zone_end) > 0 else 20
                    if tmp_diff_track < 15 and tmp_diff_end < 15:
                        rods_zone_tracking_copy.pop(i)

        # If rods are disappearing from the tracking zone, reverse the matching order.
        if len(self.rods_zone_tracking) < len(self.rods_zone_tracking_prev):
            tracking_objects_copy = dict(sorted(self.tracking_objects.items(), reverse=True))
            rods_zone_tracking_copy.reverse()

        # If the number of rods is stable, check for specific movement patterns.
        elif len(self.rods_zone_tracking) == len(self.rods_zone_tracking_prev):
            init_diff = len(self.rods_zone_init) - len(self.rods_zone_init_prev)
            diffs_tracking = [a.pos_x - b.pos_x for a, b in zip(self.rods_zone_tracking, self.rods_zone_tracking_prev)]
            mean_tracking_move = sum(diffs_tracking) / len(diffs_tracking) if diffs_tracking else 0
            end_is_stopped = end_diff < 0

            # Worst case scenario, how to solve it if there is no difference between images?
            # (Solved if there is enough difference)
            if init_diff == tracking_diff == end_diff == 0 and \
                self.rods_zone_init and self.rods_zone_tracking and self.rods_zone_end:
                self._log("ALERT: EDGE CASE I.", 100, 20*4)

            # If there are rods only in the tracking zone, then don't use associtation (Solved?)
            if (len(self.rods_zone_init) == len(self.rods_zone_end_prev) == 0) and (init_diff == end_diff == 0):
                self._log("TRYING TO SOLVE EDGE CASE II.", 100, 20*6)
                use_standard_association = False
                return tracking_objects_copy, rods_zone_tracking_copy, use_standard_association, exiting_init_zone_count

            if end_diff == 0 and self.rods_zone_end_prev:
                diffs_end = [a.pos_x - b.pos_x for a, b in zip(self.rods_zone_end, self.rods_zone_end_prev)]
                mean_end_move = sum(diffs_end) / len(diffs_end) if diffs_end else 0
                end_is_stopped = abs(mean_end_move) < 15 # Heuristic threshold

            # Special case: If tracking rods are moving right and end zone rods have stopped,
            # use a simplified, one-to-one association strategy. (Solved?)
            if mean_tracking_move >= self.cp.displacement and end_is_stopped: # Heuristic threshold
                self._log("TRYING TO SOLVE EDGE CASE I.", 100, 20*8)
                tracking_objects_copy = dict(sorted(self.tracking_objects.items(), reverse=True))
                rods_zone_tracking_copy.reverse()
                use_standard_association = False

        return tracking_objects_copy, rods_zone_tracking_copy, use_standard_association, exiting_init_zone_count

    def _associate_and_update(self,
                              objects_to_match: Dict[int, Rod],
                              rods_to_match: List[Rod],
                              use_standard_association: bool,
                              exiting_init_zone_count: int):
        """
        Matches tracked objects to current detections and updates their state.
        Handles lost tracks and creates new ones for unmatched detections.
        """
        # --- MODIFIED LOGIC for handling rods exiting the init zone ---
        if exiting_init_zone_count == 1:
            self._log(f"EXITING INIT ZONE: {exiting_init_zone_count}.", 100, 20*12)
            # Identify the 'x' leftmost rods using a temporary sorted list
            # without altering the order of the main 'rods_to_match' list.
            temp_sorted_rods = sorted(rods_to_match, key=lambda r: r.pos_x)
            newly_entered_rods = temp_sorted_rods[:exiting_init_zone_count]

            # Use a set of object IDs for efficient lookup and removal.
            newly_entered_ids = {id(rod) for rod in newly_entered_rods}
            newly_entered_rods.reverse()
            # Assign new track IDs to these newly identified rods.
            for rod in newly_entered_rods:
                rod.track_id = self.track_id
                self.tracking_objects[self.track_id] = rod
                self.track_id += 1

            # Rebuild the list, filtering out the new rods while preserving the
            # original order of the remaining rods for association.
            rods_to_match[:] = [rod for rod in rods_to_match if id(rod) not in newly_entered_ids]
        # --- END OF MODIFICATION ---

        if use_standard_association:
            self._log("USE STANDARD ASSOCIATION: TRUE.", 100, 20*10)
            unmatched_detections = deepcopy(rods_to_match)
            lost_track_ids = []

            for object_id, rod_prev in objects_to_match.items():
                found_match = False
                for rod_curr in unmatched_detections:
                    # Motion constraint: object should not move too far backward
                    if rod_curr.pos_x - rod_prev.pos_x >= self.cp.displacement: # Heuristic threshold
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
            self._log("USE STANDARD ASSOCIATION: FALSE.", 100, 20*10)
            # Simplified one-to-one association for the special "stopped" case
            for object_id, _ in objects_to_match.items():
                if rods_to_match:
                    rod_curr = rods_to_match.pop(0)
                    self.tracking_objects[object_id] = rod_curr
                    rod_curr.track_id = object_id

    def _remap_track_ids(self):
        """Ensures track IDs are consecutive, preserving their ascending order."""
        track_ids = sorted(list(self.tracking_objects.keys()), reverse=True)
        if len(track_ids) <= 1:
            return

        is_consecutive = all(track_ids[i] == track_ids[i-1] - 1 for i in range(1, len(track_ids)))
        if not is_consecutive:
            self._log(f"ALERT: REMAPPING IDS {self.tracking_objects}", 100, 20*5)

            remapped_objects = {}
            # El ID más alto se convierte en el punto de partida
            new_max_id = track_ids[0]

            for i, old_id in enumerate(track_ids):
                # El nuevo ID se calcula decrementando desde el máximo
                new_id = new_max_id - i

                # Obtenemos el objeto Rod original
                rod_object = self.tracking_objects[old_id]

                # Actualizamos el ID interno del objeto
                rod_object.track_id = new_id

                # Añadimos el objeto al nuevo diccionario con el ID corregido
                remapped_objects[new_id] = rod_object

            self.tracking_objects = dict(sorted(remapped_objects.items()))
            self.track_id = new_max_id + 1

    def _count_passing_rods(self, previous_tracks: Dict[int, Rod]):
        """Increments count for rods that cross the counting line."""
        for track_id, current_rod in self.tracking_objects.items():
            previous_rod = previous_tracks.get(track_id)
            if previous_rod and track_id not in self.counted_track_ids:
                if previous_rod.pos_x <= self.cp.counter_line and current_rod.pos_x > self.cp.counter_line:
                    self.rod_count += 1
                    self.counted_track_ids.add(track_id)

    def update_params(self, tracker_data):
        self.track_id = tracker_data['track_id']
        self.tracking_objects = tracker_data['tracking_objects']
        self.rods_zone_init_prev, self.rods_zone_tracking_prev, self.rods_zone_end_prev = self._zone_rods(tracker_data['center_points_prev_frame'])
        self.rod_count = tracker_data['rod_count']
        self.counted_track_ids = tracker_data['counted_track_ids']

    def track(self) -> Tuple[int, Dict[int, Rod], List[Rod]]:
        """
        Performs object tracking by associating current detections with existing tracks.
        """
        if self.direction == 0:
            return {'track_id': self.track_id,
                    'tracking_objects': self.tracking_objects,
                    'center_points_prev_frame': deepcopy(self.rods_cur_frame),
                    'rod_count': self.rod_count,
                    'counted_track_ids': self.counted_track_ids}

        if self.direction == 1:
            self._log(f"{self.tracking_objects}", 0, 20*14)

            # 1. If no objects are being tracked, initialize new tracks and exit.
            if not self.tracking_objects:
                self._initialize_new_tracks()
                return {'track_id': self.track_id,
                        'tracking_objects': self.tracking_objects,
                        'center_points_prev_frame': deepcopy(self.rods_cur_frame),
                        'rod_count': self.rod_count,
                        'counted_track_ids': self.counted_track_ids}

            # Store a copy of tracks before they are modified for counting later.
            previous_tracks = deepcopy(self.tracking_objects)

            # 2. Handle objects that have exited the final zone.
            self._handle_exiting_rods()

            # 3. Prepare lists for matching based on the custom heuristics.
            # This determines the strategy for associating old tracks with new detections.
            association_params = self._prepare_association_lists()

            # 4. Perform the association and update the state.
            self._associate_and_update(*association_params)

            self._remap_track_ids()

            self._log(f"{self.tracking_objects}", 0, 20*16)

            self._count_passing_rods(previous_tracks)

        return {'track_id': self.track_id,
                'tracking_objects': self.tracking_objects,
                'center_points_prev_frame': deepcopy(self.rods_cur_frame),
                'rod_count': self.rod_count,
                'counted_track_ids': self.counted_track_ids}

    def plot_count(self):
        cv2.line(self.frame, (self.cp.counter_init, 0),
                 (self.cp.counter_init, self.cp.h), self.cp.green, self.cp.font_thickness)
        cv2.line(self.frame, (self.cp.counter_end, 0),
                 (self.cp.counter_end, self.cp.h), self.cp.red, self.cp.font_thickness)
        cv2.line(self.frame, (self.cp.counter_line, 0),
                 (self.cp.counter_line, self.cp.h), self.cp.blue, self.cp.font_thickness)

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
            # Get text size to create background rectangle
            (text_width, text_height), _ = cv2.getTextSize(str(object_id), 0, 1, self.cp.font_thickness)
            # Draw white background rectangle
            cv2.rectangle(self.frame,
                         (text_pos[0] - 2, text_pos[1] - text_height - 2),
                         (text_pos[0] + text_width + 2, text_pos[1] + 2),
                         self.cp.white, -1)
            # Draw black text
            cv2.putText(self.frame, str(object_id), text_pos, 0, 1, self.cp.black, self.cp.font_thickness)

        text = f"Varillas: {self.rod_count}"

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

    def _log(self, text: str, pos_x: int = 100, pos_y: int = 20*2):
        if self.debug:
            cv2.putText(self.frame, text, (pos_x, pos_y), self.cp.font,
                        self.cp.font_scale_log, self.cp.green,
                        self.cp.font_thickness)