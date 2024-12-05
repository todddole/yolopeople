import os
import json
from tqdm import tqdm

def coco_to_yolo(coco_json_path, output_dir, image_dir, class_names):
    """
    Convert COCO annotations to YOLO format.

    Args:
        coco_json_path (str): Path to COCO JSON file.
        output_dir (str): Path to output directory.
        image_dir (str): Path to the directory containing images.
        class_names (list): List of class names to include.
    """
    # Load COCO JSON
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # Map class names to class IDs
    category_map = {cat['name']: cat['id'] for cat in coco_data['categories']}
    class_ids = {category_map[name]: idx for idx, name in enumerate(class_names)}

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    label_dir = os.path.join(output_dir, 'labels')
    os.makedirs(label_dir, exist_ok=True)
    image_output_dir = os.path.join(output_dir, 'images')
    os.makedirs(image_output_dir, exist_ok=True)

    # Process each image
    for image in tqdm(coco_data['images']):
        image_id = image['id']
        image_name = image['file_name']
        width, height = image['width'], image['height']

        # Copy image to output directory
        os.symlink(os.path.join(image_dir, image_name), os.path.join(image_output_dir, image_name))

        # Find annotations for the image
        annotations = [
            ann for ann in coco_data['annotations']
            if ann['image_id'] == image_id and ann['category_id'] in class_ids
        ]

        # Write YOLO labels
        label_path = os.path.join(label_dir, os.path.splitext(image_name)[0] + '.txt')
        with open(label_path, 'w') as label_file:
            for ann in annotations:
                bbox = ann['bbox']  # COCO bbox: [x_min, y_min, width, height]
                x_min, y_min, box_width, box_height = bbox
                x_center = (x_min + box_width / 2) / width
                y_center = (y_min + box_height / 2) / height
                norm_width = box_width / width
                norm_height = box_height / height

                # Write in YOLO format: <class_id> <x_center> <y_center> <width> <height>
                class_id = class_ids[ann['category_id']]
                label_file.write(f"{class_id} {x_center} {y_center} {norm_width} {norm_height}\n")

if __name__ == "__main__":
    # Example usage
    data_dir = "data/coco_dataset"
    class_names = ["person"]  # Add more classes if needed

    # Convert train dataset
    coco_to_yolo(
        coco_json_path=os.path.join(data_dir, "annotations/instances_train2017.json"),
        output_dir=os.path.join(data_dir, "train"),
        image_dir=os.path.join(data_dir, "train2017"),
        class_names=class_names,
    )

    # Convert validation dataset
    coco_to_yolo(
        coco_json_path=os.path.join(data_dir, "annotations/instances_val2017.json"),
        output_dir=os.path.join(data_dir, "val"),
        image_dir=os.path.join(data_dir, "val2017"),
        class_names=class_names,
    )

