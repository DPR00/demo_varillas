# This script performs a detection of the YOLO model in a given image.
from CamParameters import CameraParameters
from pathlib import Path
from torch import cuda as t_cuda
from torch import device as t_device
from ultralytics import YOLO
from utils import read_yaml_file, get_positions, plot_historic
import cv2

if __name__ == "__main__":

    dir_path = Path(__file__).resolve().parent.parent  # Absolute path of the folder
    yaml_path = dir_path/"config"/"params.yaml"
    config_data = read_yaml_file(yaml_path)

    # Get components data
    folders_data = config_data.get('folders')
    cam_data = config_data.get("camera")
    tracker_data = config_data.get("tracker")
    actuator_data = config_data.get("actuator")
    test_data = config_data.get("test")

    model_path = dir_path / folders_data.get("models") / config_data.get("model")
    image_path = dir_path / folders_data.get("imgs") / test_data.get("image") #folders_data.get("data") / config_data.get("image_test")

    # Read image
    image = cv2.imread(str(image_path))

    # Set model
    model = YOLO(str(model_path))
    device = t_device("cuda" if t_cuda.is_available() else "cpu")
    model.to(device)

    width, height, _ = image.shape
    print(width, height)
    x_init, y_init = cam_data.get("x_init"), cam_data.get("y_init")
    roi_width, roi_height = cam_data.get("roi_width"), cam_data.get("roi_height")
    cam_params = CameraParameters(width, height,
                                  x = x_init, y = y_init,
                                  w = roi_width, h = roi_height)

    min_confidence = tracker_data.get("min_confidence")

    # ROI frame
    roi_frame = image[cam_params.y : cam_params.y + cam_params.h,
                      cam_params.x : cam_params.x + cam_params.w]

    detections = model(roi_frame)
    annotated_frame = detections[0].plot()
    center_points_cur_frame, actuator_pos = get_positions(detections,
                                                          min_confidence,
                                                          actuator_data)

    plot_historic(roi_frame, [])

    for point in center_points_cur_frame:
        color = (0, 255, 0)
        cv2.circle(roi_frame, (point.pos_x, point.pos_y), 10, color, -1)

    # Show result
    # Show result
    cv2.imshow("Inference on Cropped Region", annotated_frame)
    cv2.waitKey(0)  # Wait indefinitely until a key is pressed
    cv2.imshow("Inference on Cropped Region", roi_frame)
    cv2.waitKey(0)  # Wait indefinitely until a key is pressed
    cv2.destroyAllWindows() # Close all OpenCV windows