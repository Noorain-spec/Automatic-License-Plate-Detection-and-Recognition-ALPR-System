import os
import cv2
import pandas as pd
from pathlib import Path
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split

import torch
import ultralytics
from ultralytics import YOLO

with torch.serialization.safe_globals([
    ultralytics.nn.tasks.DetectionModel,
    ultralytics.nn.modules.Conv,
    ultralytics.nn.modules.C2f,
    ultralytics.nn.modules.Detect,
]):
    model = YOLO("yolov8n.pt")

print("CUDA available:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def check_unique_extensions(path):
    extensions = set()

    for file in path.iterdir():
        if file.is_file():
            ext = file.suffix.lower()  # Normalize to lowercase
            extensions.add(ext)
    
    print("Unique file extensions found:", extensions)

def check_valid_jpg_images(image_dir: Path, df: pd.DataFrame) -> dict:
    """
    Checks if all image files listed in a DataFrame are valid JPG images.

    Args:
        image_dir (Path): The directory where the images are stored.
        df (pd.DataFrame): A DataFrame with at least two columns:
                          'images' (filename of the image) and 'labels'.

    Returns:
        dict: A dictionary containing:
              - 'all_valid' (bool): True if all images are valid, False otherwise.
              - 'invalid_images' (list): A list of filenames of invalid images.
              - 'missing_images' (list): A list of filenames of images listed
                                         in the DataFrame but not found in the directory.
    """
    invalid_images = []
    missing_images = []
    all_valid = True

    for index, row in df.iterrows():
        image_filename = row['images']
        image_path = image_dir / image_filename

        if not image_path.is_file():
            print(f"Warning: Image '{image_filename}' not found in '{image_dir}'.")
            missing_images.append(image_filename)
            all_valid = False
            continue

        try:
            with Image.open(image_path) as img:
                img.verify()  # Verify that it is an image
                if img.format != 'JPEG':
                    print(f"Warning: Image '{image_filename}' is not a JPG. Format: {img.format}")
                    invalid_images.append(image_filename)
                    all_valid = False

        except UnidentifiedImageError:
            print(f"Error: Image '{image_filename}' is not a valid image file.")
            invalid_images.append(image_filename)
            all_valid = False
        except Exception as e:
            print(f"An unexpected error occurred with image '{image_filename}': {e}")
            invalid_images.append(image_filename)
            all_valid = False

    return {
        'all_valid': all_valid,
        'invalid_images': invalid_images,
        'missing_images': missing_images
    }
# Fix errors on the dataset.yaml file

shutil.copy("input/license_plate_detection/dataset.yaml", "dataset.yaml")
fixed_yaml = """train: input/license_plate_detection/images/train
val: input/license_plate_detection/images/val

nc: 1
names: ['license_plate']
"""

with open("dataset.yaml", "w") as f:
    f.write(fixed_yaml)

lpDatasetYaml = "dataset.yaml"

# --- Setup Paths ---
# Input paths for the new dataset
lpd_base_path = Path("input/license_plate_detection")
lpd_train_images = lpd_base_path / "images/train"
lpd_val_images = lpd_base_path / "images/val"
lpd_train_labels = lpd_base_path / "labels/train"
lpd_val_labels = lpd_base_path / "labels/val"

# Output path for processed (cropped) images
lpd_img_path_processed = Path("working/lpd_images_processed")
lpd_img_path_processed.mkdir(parents=True, exist_ok=True)

# List to hold data for the DataFrame
entries_lpd = []

def process_yolo_dataset(image_dir, label_dir, output_dir):
    """
    Processes a YOLO-formatted dataset to crop and save license plates.
    """
    for label_file in os.listdir(label_dir):
        if not label_file.endswith(".txt"):
            continue

        # --- Get Image and Dimensions ---
        image_filename = label_file.replace(".txt", ".jpg")
        img_path = image_dir / image_filename
        
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Warning: Could not read image {img_path}")
            continue
        
        h, w, _ = image.shape

        # --- Read YOLO Annotation ---
        with open(label_dir / label_file, 'r') as f:
            # Process each detected plate in the image (often just one)
            for i, line in enumerate(f):
                parts = line.strip().split()
                # class_id = int(parts[0]) # '0' for license_plate
                
                # De-normalize coordinates
                x_center, y_center, width, height = map(float, parts[1:])
                x_center *= w
                y_center *= h
                width *= w
                height *= h

                # Convert from center/width/height to xtl/ytl/xbr/ybr
                xtl = int(x_center - (width / 2))
                ytl = int(y_center - (height / 2))
                xbr = int(x_center + (width / 2))
                ybr = int(y_center + (height / 2))

                # --- Crop and Save Image ---
                crop = image[ytl:ybr, xtl:xbr]
                if crop.size == 0:
                    print(f"Warning: Created an empty crop for {img_path}")
                    continue
                
                # Create a new unique filename for the cropped image
                base_name = Path(image_filename).stem
                new_filename = f"{base_name}_plate_{i}.jpg"
                new_path = output_dir / new_filename
                
                cv2.imwrite(str(new_path), crop)
                
                # As text is not available, we use a generic label
                entries_lpd.append({'images': new_filename, 'labels': 'license_plate'})

# --- Process Both Train and Validation Sets ---
print("Processing training images...")
process_yolo_dataset(lpd_train_images, lpd_train_labels, lpd_img_path_processed)

print("\nProcessing validation images...")
process_yolo_dataset(lpd_val_images, lpd_val_labels, lpd_img_path_processed)


# --- Create and Display DataFrame ---
lpd_df = pd.DataFrame(entries_lpd)
print(f"\n✅ Extracted {len(lpd_df)} cropped license plates from LPD dataset.")
lpd_df

final_lptr_df = pd.concat([lpd_df], ignore_index=True)

# lptrDataframe
ltpr_train_df, ltpr_temp_df = train_test_split(final_lptr_df, test_size=0.30, random_state=42) # 70% training data
ltpr_eval_df, ltpr_test_df = train_test_split(ltpr_temp_df, test_size=0.50, random_state=42) # 15% evaluation data and 15% testing data
print(f"\nTotal number of samples: {len(final_lptr_df)}")
print(f"Training set size: {len(ltpr_train_df)} ({len(ltpr_train_df)/len(final_lptr_df)*100:.2f}%)")
print(f"Evaluation set size: {len(ltpr_eval_df)} ({len(ltpr_eval_df)/len(final_lptr_df)*100:.2f}%)")
print(f"Test set size: {len(ltpr_test_df)} ({len(ltpr_test_df)/len(final_lptr_df)*100:.2f}%)")

#yolo_path = "input/yolov12/pytorch/default/1/yolov12n.pt"
#object_detection_model = YOLO(yolo_path)  

object_detection_model = YOLO("yolov8n.pt")
object_detection_model.train(
    data=lpDatasetYaml,
    epochs=40,                
    imgsz=640,                
    batch=16,                 
    workers=2,
    lr0= 0.0005,
    lrf=0.1,              
    patience=5,              
    save=True,                
    name="yolov12n-license-plate",
    augment=True
)
