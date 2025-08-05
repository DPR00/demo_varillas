from scripts import CameraParameters, Logger, Tracker, plot_historic, get_positions, get_data, handle_actuator
import os
from torch import cuda as t_cuda
from torch import device as t_device
from ultralytics import YOLO
import cv2
import time

take_time = False

if __name__ == "__main__":
    # timestamp_string = time.strftime("%Y-%m-%d %H:%M:%S", current_struct_time)
    if take_time:
        start_time = time.perf_counter()

    dir_path = os.path.dirname(os.path.abspath(__file__))
    data = get_data(dir_path)

    if take_time:
        end_time = time.perf_counter()
        elapsed_ms = (end_time - start_time)*1000
        print(f"Init time: {elapsed_ms:.2f}")

        start_time = time.perf_counter()

    # Open the video file
    cap = cv2.VideoCapture(data['video_path'])
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)        # Mantén solo 1 frame en el buffer
    cap.set(cv2.CAP_PROP_FPS, 30)              # Ajusta al FPS real de tu cámara
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H','2','6','4'))  # Códec H.264

    # Si tu OpenCV lo soporta:
    # cap.set(cv2.CAP_PROP_RTSP_TRANSPORT, 1)    # 0=Any, 1=UDP, 2=TCP
    # cap.set(cv2.CAP_PROP_LATENCY, 0)           # Forzar latencia mínima

    # Check if the video file opened successfully
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir el video o stream: {data['video_path']}")
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
                                  x = data['x_init'], y = data['y_init'],
                                  w = data['roi_width'], h = data['roi_height'])
    cam_params.update_limits(data['counter_init'], data['counter_end'], data['counter_line'])

    # Variables
    frame_count = 0
    tracker_data = {'track_id': 1,
                    'tracking_objects': {},
                    'rod_count': 0,
                    'counted_track_ids': set(),
                    'center_points_prev_frame': []}
    list_counter = [] # For plot historic
    actuator_initial_pos = (0,0)
    prev_size = -1
    actuator_moving = False
    store_package = False
    direction = 1 # 1: left to right (Default), 0: stop, -1: right to left
    actuactor_count = 0

    if take_time:
        end_time = time.perf_counter()
        elapsed_ms = (end_time - start_time)*1000
        print(f"Set variable time: {elapsed_ms:.2f}")

        start_time = time.perf_counter()
    # Set model
    model = YOLO(data['model_path'])
    device = t_device("cuda" if t_cuda.is_available() else "cpu")
    model.to(device)

    if take_time:
        end_time = time.perf_counter()
        elapsed_ms = (end_time - start_time)*1000
        print(f"Set model time: {elapsed_ms:.2f}")

    # Video writer
    if data['generate_video']:
        video_writer = cv2.VideoWriter(data['output_path'], \
                                       cv2.VideoWriter_fourcc(*'mp4v'), \
                                       fps, \
                                       (data['roi_width'], data['roi_height']))
        if not video_writer.isOpened():
            print(f"[ERROR] Could not initialize video writer for {data['output_path']}")
            data['generate_video'] = False
    # Logger
    storage_path = data['storage_path'] if data['storage_data'] else None
    logger = Logger(output_dir = data['logger_path'], storage_path = storage_path)

    while cap.isOpened():
        if take_time:
            print(f"---------- Frame: {frame_count} ------------------")
            start_time_global = time.perf_counter()

        if cv2.waitKey(1) & 0xFF == ord('p'):  # Press 'q' to exit
            actuator_moving = not actuator_moving

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
            plot_historic(roi_frame, list_counter, data['logo'])

        center_points_cur_frame, actuator_pos = get_positions(detections,
                                                              data['min_confidence'],
                                                              data['actuator_data'])

        if actuator_pos[1] != 0 and actuator_pos[1] != 0:
            cv2.circle(roi_frame, (actuator_pos[0], actuator_pos[1]), 10, (0,0,255), -1)
        if data['debug']:
            cv2.putText(roi_frame, f"Apos: {actuator_pos}", (50, 20*9), cam_params.font, cam_params.font_scale, cam_params.green, cam_params.font_thickness*2)
        sorted_center_points_cur_frame = sorted(center_points_cur_frame, key = lambda point: point.pos_x)
        sorted_center_points_cur_frame.reverse()
        list_counter, tracker_data, store_package, actuactor_count = handle_actuator(cam_params, actuator_pos, list_counter, tracker_data, store_package, actuactor_count)

        if not actuator_moving:
            # print(frame_count+1, end=". ")
            tracker = Tracker(sorted_center_points_cur_frame, roi_frame, cam_params, debug=data['debug'], direction=direction)
            tracker.update_params(tracker_data)
            tracker_data = tracker.track()
            tracker.plot_count()
            store_package = False
            actuactor_count = 0

        if take_time:
            end_time = time.perf_counter()
            elapsed_ms = (end_time - start_time)*1000
            print(f"Post processing time: {elapsed_ms:.2f} ms.")

        frame_count += 1
        prev_size = len(list_counter)

        if frame_count == 1:
            actuator_initial_pos = actuator_pos

        if data['generate_video']:
            video_writer.write(roi_frame)

        if data['debug']:
            logger.log(roi_frame, frame_count)

        if data['storage_data']:
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
    cap.release()
    if data['generate_video']:
        print(f"Processing complete. Video saved to {data['output_path']}")
        video_writer.release()
    cv2.destroyAllWindows()
