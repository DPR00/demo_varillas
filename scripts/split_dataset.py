#!/usr/bin/env python3
"""
Script to split dataset into train and validation folders.
Splits PNG images and their corresponding TXT label files from dataset/dataset_varillas
into train (70%) and val (30%) folders with the specified structure.
"""

import os
import shutil
import random
from pathlib import Path
import argparse


def create_directory_structure(base_path):
    """Create the required directory structure."""
    directories = [
        base_path / "train" / "images",
        base_path / "train" / "labels", 
        base_path / "val" / "images",
        base_path / "val" / "labels"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")


def get_matching_files(images_dir, labels_dir):
    """Get pairs of image and label files that exist in both directories."""
    image_files = set(f.stem for f in images_dir.glob("*.png"))
    label_files = set(f.stem for f in labels_dir.glob("*.txt"))
    
    # Find files that have both image and label
    matching_files = image_files.intersection(label_files)
    
    print(f"Found {len(matching_files)} matching image-label pairs")
    return sorted(list(matching_files))


def split_dataset(dataset_path, train_ratio=0.7, seed=42):
    """
    Split dataset into train and validation sets.
    
    Args:
        dataset_path: Path to the dataset directory
        train_ratio: Ratio of files to use for training (default: 0.7)
        seed: Random seed for reproducibility
    """
    dataset_path = Path(dataset_path)
    images_dir = dataset_path / "images"
    labels_dir = dataset_path / "labels"
    
    # Check if source directories exist
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")
    
    # Create directory structure
    create_directory_structure(dataset_path)
    
    # Get matching files
    matching_files = get_matching_files(images_dir, labels_dir)
    
    if not matching_files:
        print("No matching image-label pairs found!")
        return
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Shuffle files
    random.shuffle(matching_files)
    
    # Calculate split indices
    num_train = int(len(matching_files) * train_ratio)
    train_files = matching_files[:num_train]
    val_files = matching_files[num_train:]
    
    print(f"Training files: {len(train_files)}")
    print(f"Validation files: {len(val_files)}")
    
    # Move files to appropriate directories
    for filename in train_files:
        # Move image
        src_image = images_dir / f"{filename}.png"
        dst_image = dataset_path / "train" / "images" / f"{filename}.png"
        shutil.copy2(src_image, dst_image)
        
        # Move label
        src_label = labels_dir / f"{filename}.txt"
        dst_label = dataset_path / "train" / "labels" / f"{filename}.txt"
        shutil.copy2(src_label, dst_label)
    
    for filename in val_files:
        # Move image
        src_image = images_dir / f"{filename}.png"
        dst_image = dataset_path / "val" / "images" / f"{filename}.png"
        shutil.copy2(src_image, dst_image)
        
        # Move label
        src_label = labels_dir / f"{filename}.txt"
        dst_label = dataset_path / "val" / "labels" / f"{filename}.txt"
        shutil.copy2(src_label, dst_label)
    
    print(f"\nDataset split completed!")
    print(f"Training set: {len(train_files)} files")
    print(f"Validation set: {len(val_files)} files")
    print(f"Total files processed: {len(matching_files)}")


def main():
    parser = argparse.ArgumentParser(description="Split dataset into train and validation sets")
    parser.add_argument(
        "--dataset-path", 
        default="dataset/dataset_varillas",
        help="Path to the dataset directory (default: dataset/dataset_varillas)"
    )
    parser.add_argument(
        "--train-ratio", 
        type=float, 
        default=0.7,
        help="Ratio of files to use for training (default: 0.7)"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    try:
        split_dataset(args.dataset_path, args.train_ratio, args.seed)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 