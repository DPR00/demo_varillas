import os
import cv2

class Logger:
    def __init__(self, output_dir = "./../imgs/imgs-main_app", storage_path = None):
        self.output_dir_result = output_dir
        self.image_prefix = "frame"
        os.makedirs(self.output_dir_result, exist_ok=True)
        self.storage_path = storage_path
        if storage_path is not None:
            os.makedirs(storage_path, exist_ok=True)


    def log(self, resized_frame, frame_count):
        # Construct the filename for the current frame
        image_filename = f"{self.image_prefix}_{frame_count:04d}.jpg"
        image_path = os.path.join(self.output_dir_result, image_filename)
        cv2.imwrite(image_path, resized_frame)

    def save_img(self, resized_frame, frame_count):
        if self.storage_path is not None:
            image_filename = f"{self.image_prefix}_{frame_count:04d}.jpg"
            image_path = os.path.join(self.storage_path, image_filename)
            cv2.imwrite(image_path, resized_frame)
        else:
            print("ERROR: Image was not saved. Make sure storage path is set.")