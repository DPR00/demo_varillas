from tarfile import data_filter
import cv2
import numpy as np
import time
import yaml
import os
from .datatypes import Rod
from .CamParameters import CameraParameters

def read_yaml_file(path: str):
    try:
        # Open the file and load the YAML data
        with open(path, 'r') as file:
            config_data = yaml.safe_load(file)

        return config_data

    except FileNotFoundError:
        print(f"Error: The file '{path}' was not found.")
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")

def get_positions(detections, confidence, actuator_data):
    center_points_cur_frame = []
    actuator_poses = []
    x_offset =  actuator_data.get("x_offset")
    y_limit = actuator_data.get("y_limit")
    actuator_pos = (0,0)

    for detection in detections:
        boxes = detection.boxes.xyxy.cpu().numpy()  # Convert tensors to NumPy
        confidences = detection.boxes.conf.cpu().numpy()
        cls_arr = detection.boxes.cls.cpu().numpy()

        # Filter boxes based on confidence > 0.8
        valid_mask = confidences > confidence
        valid_boxes = boxes[valid_mask]
        valid_cls = cls_arr[valid_mask]

        # Identify the index where class == 1
        id_sep = np.where(valid_cls == 1)[0]

        # Compute centers efficiently
        centers = valid_boxes[:, :2] + valid_boxes[:, 2:]  # x1 + x2, y1 + y2
        centers = (centers / 2).astype(int)  # Compute (center_x, center_y)
        # Separate actuator vs. other points
        for i, (center_x, center_y) in enumerate(centers):
            if i in id_sep:
                actuator_poses.append((center_x + x_offset, center_y)) # ORIGINAL +100
            else:
                rod = Rod(track_id = -1, pos_x = center_x, pos_y = center_y)
                center_points_cur_frame.append(rod)

    if len(actuator_poses)!=0:
        actuator_pos = min(actuator_poses, key=lambda item: item[1])
    return center_points_cur_frame, actuator_pos

def get_data(dir_path):
    data = {}

    current_struct_time = time.localtime()
    timestamp_string = time.strftime("%Y-%m-%d", current_struct_time)
    # Absolute path of the folder two levels up from the current script
    yaml_path = os.path.join(dir_path, 'config', 'params.yaml')
    config_data = read_yaml_file(yaml_path)
    folders_data = config_data.get('folders')
    cam_data = config_data.get("camera")
    tracker_data = config_data.get("tracker")
    actuator_data = config_data.get("actuator")
    # Get paths using os.path.join for cross-platform compatibility
    input_video = config_data.get("input_video")
    logo_path = os.path.join(dir_path, folders_data.get("assets"), config_data.get("logo"))
    logger_path = os.path.join(dir_path, folders_data.get("logger"), config_data.get("version"))
    output_path = os.path.join(dir_path, folders_data.get("output"), config_data.get("version") + timestamp_string + ".mp4")
    storage_path = os.path.join(dir_path, folders_data.get("storage"), config_data.get("version"))
    model_path = os.path.join(dir_path, folders_data.get("models"), config_data.get("model"))
    if input_video and input_video.startswith("rtsp://"):
        video_path = input_video
    else:
        video_path = os.path.join(dir_path, folders_data.get("media"), input_video)

    debug = config_data.get("debug_mode")
    generate_video = config_data.get("generate_video")
    min_confidence = tracker_data.get("min_confidence")
    x_init, y_init = cam_data.get("x_init"), cam_data.get("y_init")
    roi_width, roi_height = cam_data.get("roi_width"), cam_data.get("roi_height")
    counter_init, counter_end, counter_line = cam_data.get("counter_init"), cam_data.get("counter_end"), cam_data.get("counter_line")
    plot_x_offset, plot_y_offset = cam_data.get("plot_x_offset"), cam_data.get("plot_y_offset")
    act_y_init, act_y_finish = actuator_data.get("y_init"), actuator_data.get("y_finish")
    storage_data = config_data.get("storage_data")

    logo = cv2.imread(logo_path)

    data["actuator_data"] = actuator_data
    data["model_path"] = model_path
    data["video_path"] = video_path
    data["logo"] = logo
    data["logger_path"] = logger_path
    data["output_path"] = output_path
    data["storage_path"] = storage_path
    data["debug"] = debug
    data["generate_video"] = generate_video
    data["min_confidence"] = min_confidence
    data["x_init"] = x_init
    data["y_init"] = y_init
    data["roi_width"] = roi_width
    data["roi_height"] = roi_height
    data["counter_init"] = counter_init
    data["counter_end"] = counter_end
    data["counter_line"] = counter_line
    data["plot_x_offset"] = plot_x_offset
    data["plot_y_offset"] = plot_y_offset
    data["act_y_init"] = act_y_init
    data["act_y_finish"] = act_y_finish
    data["storage_data"] = storage_data

    return data

def plot_historic(main_image, list_counter, logo):
    paquetes = list_counter.copy()
    # If the logo has an alpha channel, convert it to BGR
    if logo.shape[2] == 4:
        logo = cv2.cvtColor(logo, cv2.COLOR_BGRA2BGR)  # Remove alpha channel

    # Resize the logo to 10% of its original size
    scale_percent = 10  # Resize to 10% of original size
    new_width = int(logo.shape[1] * scale_percent / 100)
    new_height = int(logo.shape[0] * scale_percent / 100)
    logo_resized = cv2.resize(logo, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Get dimensions
    h_main, w_main, _ = main_image.shape
    h_logo, w_logo, _ = logo_resized.shape

    # Define bottom-right position
    x_offset = 10
    y_offset = 10

    # Overlay the logo by direct assignment
    main_image[y_offset:y_offset + h_logo, x_offset:x_offset + w_logo] = logo_resized

    # Define text parameters
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    color = (255, 255, 255)  # Black color
    line_height = 28  # Adjust line spacing

    # Initial Y position for text
    x_start = 95
    y_start = 30

    # Write the title
    cv2.putText(main_image, "Historico general", (x_start, y_start), font, font_scale, (0,255,0), thickness)

    start_package_number = 1  # This will increase when we remove elements

    # Add package data if the list is not empty
    y_position = y_start + line_height

    if len(paquetes) == 0:
        text_lines = []
    else:
        while paquetes:

            text_lines = [f"Paquete {start_package_number + i}: {paquetes[i]}" for i in range(len(paquetes))]

            # Check if text fits in the image height
            if y_position + (len(text_lines) * line_height) > h_main//3:
                paquetes.pop(0)  # Remove the first element if it overflows
                start_package_number += 1  # Increase the starting number
            else:
                break  # If it fits, stop removing elements
    # Draw the remaining text
    for i, text in enumerate(text_lines):
        cv2.putText(main_image, text, (x_start, y_position), font, font_scale, color, thickness)
        y_position += line_height
