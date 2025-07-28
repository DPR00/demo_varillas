import os
from PIL import Image
import shutil

# --- Configuration ---
# Define the dimensions and crop coordinates
ORIG_W, ORIG_H = 3840, 2160
CROP_W, CROP_H = 1548, 1105

# The (x, y) coordinates for the top-left corner of each of the 6 crops
CROP_COORDS = [
    (0, 0),         # Top-left
    (1146, 0),      # Top-middle
    (2292, 0),      # Top-right
    (0, 1055),      # Bottom-left
    (1146, 1055),   # Bottom-middle
    (2292, 1055)    # Bottom-right
]

# Base folder names
SOURCE_IMG_DIR = 'images'
SOURCE_LBL_DIR = 'labels'
DEST_IMG_DIR = 'images_rs'
DEST_LBL_DIR = 'labels_rs'
EMPTY_IMG_DIR = 'empty_images_rs' # New folder for images without labels


def process_and_tile_dataset(source_directory):
    """
    Main function to iterate through 'train' and 'val' sets,
    tiling images and transforming their corresponding labels.
    """
    print("🚀 Starting dataset tiling process...")

    # Create the top-level directory for empty images
    os.makedirs(EMPTY_IMG_DIR, exist_ok=True)

    # Iterate through both training and validation sets
    for split in ['train', 'val']:
        print(f"\n--- Processing '{split}' set ---")

        # Define source and destination paths for the current split
        src_img_path = os.path.join(source_directory, SOURCE_IMG_DIR, split)
        src_lbl_path = os.path.join(source_directory, SOURCE_LBL_DIR, split)
        dest_img_path = os.path.join(source_directory, DEST_IMG_DIR, split)
        dest_lbl_path = os.path.join(source_directory, DEST_LBL_DIR, split)

        # Create destination directories if they don't exist
        os.makedirs(dest_img_path, exist_ok=True)
        os.makedirs(dest_lbl_path, exist_ok=True)

        # Get list of images to process
        if not os.path.exists(src_img_path):
            print(f"⚠️ Warning: Source image directory not found: {src_img_path}")
            continue

        image_files = [f for f in os.listdir(src_img_path) if f.endswith(('.png', '.jpg'))]

        for i, filename in enumerate(image_files):
            base_filename = os.path.splitext(filename)[0]
            print(f"  > Processing image {i+1}/{len(image_files)}: {filename}")

            original_image = Image.open(os.path.join(src_img_path, filename))
            original_label_path = os.path.join(src_lbl_path, f"{base_filename}.txt")

            original_boxes = []
            if os.path.exists(original_label_path):
                with open(original_label_path, 'r') as f:
                    original_boxes = [line.strip().split() for line in f.readlines()]

            for crop_idx, (crop_x, crop_y) in enumerate(CROP_COORDS):
                new_filename_base = f"{base_filename}_crop{crop_idx + 1}"

                # First, determine the new labels for the crop
                new_labels = []
                for box in original_boxes:
                    class_id, x_center, y_center, width, height = map(float, box)
                    abs_x_center = x_center * ORIG_W
                    abs_y_center = y_center * ORIG_H

                    if (crop_x <= abs_x_center < crop_x + CROP_W) and \
                       (crop_y <= abs_y_center < crop_y + CROP_H):

                        abs_width = width * ORIG_W
                        abs_height = height * ORIG_H
                        new_abs_x_center = abs_x_center - crop_x
                        new_abs_y_center = abs_y_center - crop_y

                        new_norm_x = min(max(new_abs_x_center / CROP_W, 0), 1)
                        new_norm_y = min(max(new_abs_y_center / CROP_H, 0), 1)
                        new_norm_w = min(max(abs_width / CROP_W, 0), 1)
                        new_norm_h = min(max(abs_height / CROP_H, 0), 1)

                        new_labels.append(f"{int(class_id)} {new_norm_x:.6f} {new_norm_y:.6f} {new_norm_w:.6f} {new_norm_h:.6f}")

                # Now, crop the image
                crop_box = (crop_x, crop_y, crop_x + CROP_W, crop_y + CROP_H)
                cropped_image = original_image.crop(crop_box)

                # Conditionally save the image and label based on whether labels exist
                if new_labels:
                    # Save image to the main dataset folder
                    cropped_image.save(os.path.join(dest_img_path, f"{new_filename_base}.png"))
                    # Save the new labels
                    with open(os.path.join(dest_lbl_path, f"{new_filename_base}.txt"), 'w') as f:
                        f.write('\n'.join(new_labels))
                else:
                    # If no labels, save image to the 'empty_images_rs' folder
                    cropped_image.save(os.path.join(EMPTY_IMG_DIR, f"{new_filename_base}.png"))

    print("\n✅ Tiling process completed successfully!")


if __name__ == "__main__":
    try:
        # Get the absolute path of the script being run
        script_path = os.path.abspath(__file__)
        # Get the directory containing the script ('scripts' folder)
        scripts_dir = os.path.dirname(script_path)
        # Get the parent directory of 'scripts' (the project root)
        project_root = os.path.dirname(scripts_dir)
    except NameError:
        # Fallback for environments where __file__ is not defined (e.g., interactive)
        project_root = os.path.abspath('.')
        print("⚠️ Warning: '__file__' not found. Assuming script is run from 'scripts' directory.")

    source_directory = os.path.join(project_root, 'dataset')

    process_and_tile_dataset(source_directory)