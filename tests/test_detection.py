import unittest
import json
import os
from ultralytics import YOLO

# --- Main Test Class ---
class TestYoloDetections(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up the test environment before any tests run."""
        print("Setting up test environment...")

        # --- Configuration ---
        cls.MODEL_PATH = 'path/to/your/best.pt'
        cls.IMAGES_DIR = 'path/to/your/test_images/'
        cls.EXPECTED_JSON_PATH = 'expected_outputs.json'
        cls.CONF_THRESHOLD = 0.25
        cls.IOU_THRESHOLD = 0.5

        # Load the YOLO model
        cls.model = YOLO(cls.MODEL_PATH)

        # Load the expected outputs from the JSON file
        with open(cls.EXPECTED_JSON_PATH, 'r') as f:
            cls.expected_data = json.load(f)

    def test_object_detection(self):
        """Iterates through test images and verifies detections."""

        for image_name, expected_objects in self.expected_data.items():
            with self.subTest(image=image_name):
                image_path = os.path.join(self.IMAGES_DIR, image_name)
                self.assertTrue(os.path.exists(image_path), f"Image not found: {image_path}")

                # Run inference
                results = self.model(image_path, conf=self.CONF_THRESHOLD, verbose=False)

                # Get detected boxes
                pred_boxes = results[0].boxes.xywhn.cpu().numpy().tolist()
                pred_classes = results[0].boxes.cls.cpu().numpy().tolist()

                # 1. Test the number of detected objects
                self.assertEqual(len(pred_boxes), len(expected_objects), f"Incorrect number of detections in {image_name}")

                if not expected_objects:
                    continue # Test passes if both are empty

                # 2. Match and test each expected object
                unmatched_preds = list(range(len(pred_boxes)))

                for expected_obj in expected_objects:
                    expected_class_id = expected_obj['class_id']
                    best_match_idx = -1

                    # Assert the class of the matched box is correct
                    matched_pred_class = pred_classes[best_match_idx]
                    self.assertEqual(matched_pred_class, expected_class_id, f"Class mismatch in {image_name}")

                    # Remove the matched prediction so it can't be used again
                    unmatched_preds.remove(best_match_idx)

if __name__ == '__main__':
    unittest.main()