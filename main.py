from scripts import CameraParameters, Logger, Tracker, plot_historic, read_yaml_file, get_positions
import os
from torch import cuda as t_cuda
from torch import device as t_device
from ultralytics import YOLO
import cv2
import time

take_time = False

if __name__ == "__main__":
    if take_time:
        start_time = time.perf_counter()

    # Absolute path of the folder two levels up from the current script
    dir_path = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(dir_path, 'config', 'params.yaml')
    config_data = read_yaml_file(yaml_path)
    folders_data = config_data.get('folders')

    # Get paths using os.path.join for cross-platform compatibility
    model_path = os.path.join(dir_path, folders_data.get("models"), config_data.get("model"))
    video_path = os.path.join(dir_path, folders_data.get("media"), config_data.get("input_video"))
    logo_path = os.path.join(dir_path, folders_data.get("assets"), config_data.get("logo"))
    logger_path = os.path.join(dir_path, folders_data.get("logger"), config_data.get("version"))
    output_path = os.path.join(dir_path, folders_data.get("output"), config_data.get("version") + ".mp4")
    storage_path = os.path.join(dir_path, folders_data.get("storage"), config_data.get("version"))

    # Get components data
    cam_data = config_data.get("camera")
    tracker_data = config_data.get("tracker")
    actuator_data = config_data.get("actuator")

    # Get individual data
    debug = config_data.get("debug_mode")
    generate_video = config_data.get("generate_video")
    min_confidence = tracker_data.get("min_confidence")
    x_init, y_init = cam_data.get("x_init"), cam_data.get("y_init")
    roi_width, roi_height = cam_data.get("roi_width"), cam_data.get("roi_height")
    counter_init, counter_end = cam_data.get("counter_init"), cam_data.get("counter_end")
    plot_x_offset, plot_y_offset = cam_data.get("plot_x_offset"), cam_data.get("plot_y_offset")
    act_y_init, act_y_finish = actuator_data.get("y_init"), actuator_data.get("y_finish")
    storage_data = config_data.get("storage_data")

    logo = cv2.imread(logo_path)  # Keep transparency if present

    if take_time:
        end_time = time.perf_counter()
        elapsed_ms = (end_time - start_time)*1000
        print(f"Init time: {elapsed_ms:.2f}")

        start_time = time.perf_counter()

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video file opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    if take_time:
        end_time = time.perf_counter()
        elapsed_ms = (end_time - start_time)*1000
        print(f"Open capture time: {elapsed_ms:.2f}")

        start_time = time.perf_counter()

    # Video parameters
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cam_params = CameraParameters(width, height,
                                  x = x_init, y = y_init,
                                  w = roi_width, h = roi_height)
    cam_params.update_limits(counter_init, counter_end)

    # Variables
    frame_count = 0
    track_id = 1
    tracking_objects = {}
    center_points_prev_frame = []
    list_counter = [] # For plot historic
    actuator_initial_pos = (0,0)
    min_track = 0
    prev_size = -1
    max_key = -1
    counted = False
    stored_list = False
    actuator_moving = False

    if take_time:
        end_time = time.perf_counter()
        elapsed_ms = (end_time - start_time)*1000
        print(f"Set variable time: {elapsed_ms:.2f}")

        start_time = time.perf_counter()
    # Set model
    model = YOLO(model_path)
    device = t_device("cuda" if t_cuda.is_available() else "cpu")
    model.to(device)

    if take_time:
        end_time = time.perf_counter()
        elapsed_ms = (end_time - start_time)*1000
        print(f"Set model time: {elapsed_ms:.2f}")

    # Video writer
    if generate_video:
        video_writer = cv2.VideoWriter(output_path,
                                       cv2.VideoWriter_fourcc(*'mp4v'),
                                       fps,
                                       (roi_width, roi_height))

    # Logger
    storage_path = storage_path if storage_data else None
    logger = Logger(output_dir = logger_path, storage_path = storage_path)

    while cap.isOpened():
        if take_time:
            print(f"---------- Frame: {frame_count} ------------------")
            start_time_global = time.perf_counter()

        success, frame = cap.read()

        if not success:
            print("No frame.")
            break

        if take_time:
            end_time = time.perf_counter()
            elapsed_ms = (end_time - start_time_global)*1000
            print(f"Reading time: {elapsed_ms:.2f} ms.")

            start_time = time.perf_counter()
        # ROI frame
        roi_frame = frame[cam_params.y : cam_params.y + cam_params.h,
                          cam_params.x : cam_params.x + cam_params.w]
        clean_roi_frame = roi_frame.copy()

        detections = model(roi_frame, verbose=False)

        if take_time:
            end_time = time.perf_counter()
            elapsed_ms = (end_time - start_time)*1000
            print(f"Detection time: {elapsed_ms:.2f} ms.")

            start_time = time.perf_counter()

        if prev_size == len(list_counter):
            plot_historic(roi_frame, list_counter, logo)

        center_points_cur_frame, actuator_pos = get_positions(detections,
                                                              min_confidence,
                                                              actuator_data)
        # cv2.circle(roi_frame, (actuator_pos[0], actuator_pos[1]), 10, (0,0,255), -1)

        sorted_center_points_cur_frame = sorted(center_points_cur_frame, key = lambda point: point.pos_x)
        sorted_center_points_cur_frame = sorted_center_points_cur_frame[::-1]
        actuator_moving = False #  actuator_pos[1]!= 0 \
                           #and (abs(actuator_pos[1] - actuator_initial_pos[1]) > act_y_init) \
                           #and frame_count > 1

        # # Plot counting if actuator start moving
        # if actuator_moving:
        #     # Check if actuator finished moving
        #     if len(tracking_objects) > 0 and actuator_pos[1] <= act_y_finish and (not counted):
        #         track_id = 1
        #         # print(tracking_objects)
        #         point_max = max([point for _, point in tracking_objects.items() if point.pos_x <= actuator_pos[0]], key = lambda point : point.pos_x)
        #         # print(point_max, actuator_pos[0])
        #         max_key = point_max.track_id
        #         counted = True

        #     if (max_key != -1):
        #         cv2.putText(roi_frame, "Ingresaron " + str(max_key) + " varillas",
        #                     (int(cam_params.w//2 - plot_x_offset), int(cam_params.y//2 + plot_y_offset)),
        #                     cam_params.font, cam_params.font_scale*3, cam_params.text_color, cam_params.font_thickness*2)
        #         if not stored_list:
        #             list_counter.append(max_key)
        #             stored_list = True
        # else:
        #     stored_list = False
        #     counted = False

        if not actuator_moving:
            # print(frame_count+1, end=". ")
            tracker = Tracker(sorted_center_points_cur_frame, roi_frame, cam_params, debug=debug)
            tracker.update_params(track_id, tracking_objects, center_points_prev_frame)
            track_id, tracking_objects, center_points_prev_frame = tracker.track()
            tracker.plot_count()

        if take_time:

            end_time = time.perf_counter()
            elapsed_ms = (end_time - start_time)*1000
            print(f"Post processing time: {elapsed_ms:.2f} ms.")

        frame_count += 1
        prev_size = len(list_counter)

        if frame_count == 1:
            actuator_initial_pos = actuator_pos

        if generate_video:
            video_writer.write(roi_frame)

        if debug:
            logger.log(roi_frame, frame_count)

        if storage_data:
            logger.save_img(clean_roi_frame, frame_count)

        if take_time:
            start_time = time.perf_counter()

        # Show result
        cv2.imshow("Inference on Cropped Region", roi_frame)

        if take_time:
            end_time = time.perf_counter()
            elapsed_ms = (end_time - start_time)*1000
            print(f"CV2 imshow time: {elapsed_ms:.2f}")
            elapsed_ms = (end_time - start_time_global)*1000
            print(f"Total time: {elapsed_ms:.2f} ms.")

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    # 6. Release resources
    print(f"Processing complete. Video saved to {str(output_path)}")
    cap.release()
    if generate_video:
        video_writer.release()
    cv2.destroyAllWindows()
