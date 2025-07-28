import os
import shutil
import random
import glob
import sys

def organize_yolo_dataset(source_dir: str, output_dir: str, split_ratio: float = 0.8):
    """
    Organizes image and label files from a source directory into a YOLO-compatible
    format, renaming files and splitting them into training and validation sets.

    Args:
        source_dir (str): The absolute path to the 'dataset' directory containing 'GR*' folders.
        output_dir (str): The absolute path where the structured dataset folders will be created.
        split_ratio (float): The ratio of files to be used for the training set.
    """
    # --- 1. Create Destination Directories ---
    print("🚀 Starting the organization process...")
    paths = {
        "img_train": os.path.join(output_dir, 'images', 'train'),
        "img_val": os.path.join(output_dir, 'images', 'val'),
        "lbl_train": os.path.join(output_dir, 'labels', 'train'),
        "lbl_val": os.path.join(output_dir, 'labels', 'val'),
        "empty": os.path.join(output_dir, 'empty_images')
    }
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    print("✅ Destination directories are ready.")

    # --- 2. Find and Process Files ---
    valid_files = []
    empty_image_count = 0

    # Find all 'GR*' prefixed folders in the source directory
    main_folders = glob.glob(os.path.join(source_dir, 'GR*'))
    if not main_folders:
        sys.exit(f"Error: No 'GR*' folders found in '{source_dir}'. Please check the path and folder structure.")

    print(f"🔍 Found {len(main_folders)} source folders. Processing files...")

    for folder_path in main_folders:
        if not os.path.isdir(folder_path): continue # Skip if it's not a directory

        folder_name = os.path.basename(folder_path)
        prefix = folder_name[:12]

        # Dynamically find image and label subdirectories
        img_subdir, lbl_subdir = None, None
        for sub_dir in os.listdir(folder_path):
            sub_dir_path = os.path.join(folder_path, sub_dir)
            if os.path.isdir(sub_dir_path):
                if glob.glob(os.path.join(sub_dir_path, '*.png')):
                    img_subdir = sub_dir_path
                elif glob.glob(os.path.join(sub_dir_path, '*.txt')):
                    lbl_subdir = sub_dir_path

        if not img_subdir:
            print(f"⚠️ Warning: No image subdirectory with .png files found in '{folder_path}'. Skipping.")
            continue

        # Process each image in the found image subdirectory
        for img_path in glob.glob(os.path.join(img_subdir, '*.png')):
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            new_img_name = f"{prefix}_{os.path.basename(img_path)}"

            # Assume label file is in the label subdir, if found
            lbl_path = None
            if lbl_subdir:
                lbl_path = os.path.join(lbl_subdir, f"{base_name}.txt")

            # Check if the corresponding label file exists
            if lbl_path and os.path.exists(lbl_path):
                new_lbl_name = f"{prefix}_{base_name}.txt"
                valid_files.append((img_path, lbl_path, new_img_name, new_lbl_name))
            else:
                # If no label, copy image to 'empty_images'
                dest_path = os.path.join(paths["empty"], new_img_name)
                shutil.copy(img_path, dest_path)
                empty_image_count += 1

    # --- 3. Shuffle and Split Data ---
    random.shuffle(valid_files)
    split_index = int(len(valid_files) * split_ratio)
    train_set = valid_files[:split_index]
    val_set = valid_files[split_index:]
    print("🔀 Shuffling and splitting data into training and validation sets.")

    # --- 4. Copy Files to Final Destination ---
    def copy_files(file_set, img_dest_path, lbl_dest_path):
        for src_img, src_lbl, new_img, new_lbl in file_set:
            shutil.copy(src_img, os.path.join(img_dest_path, new_img))
            shutil.copy(src_lbl, os.path.join(lbl_dest_path, new_lbl))

    copy_files(train_set, paths["img_train"], paths["lbl_train"])
    copy_files(val_set, paths["img_val"], paths["lbl_val"])

    # --- 5. Final Summary ---
    print("\n" + "="*40)
    print("🎉 Organization Complete! 🎉")
    print("="*40)
    print(f"Total processed pairs (image + label): {len(valid_files)}")
    print(f"  - 🚂 Training set: {len(train_set)} files")
    print(f"  - 🧪 Validation set: {len(val_set)} files")
    print(f"Images without labels (in 'empty_images'): {empty_image_count}")
    print(f"Your dataset is ready at: '{output_dir}'")
    print("="*40)


if __name__ == '__main__':
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

    # Define the source and output directories using absolute paths
    # Source is the 'dataset' folder containing the 'GR*' folders
    source_directory = os.path.join(project_root, 'dataset')

    # Output folders ('images', 'labels') will also be created inside the 'dataset' folder
    output_directory = source_directory

    # Verify that the dataset directory exists before running
    if not os.path.isdir(source_directory):
        print(f"❌ Error: The directory '{source_directory}' could not be found.")
        print("Please ensure your folder structure is correct:")
        print("project_root/\n|-- dataset/         <-- (must exist)\n|-- scripts/\n    |-- make_training.py")
        sys.exit(1) # Exit the script if the path is wrong

    organize_yolo_dataset(source_directory, output_directory)