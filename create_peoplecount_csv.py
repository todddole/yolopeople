import json
import csv
import os

def create_people_count_csv(coco_json_path, image_dir, output_csv_path):
    """
    Create a CSV file with image IDs, names, and the number of people in each image.

    Args:
        coco_json_path (str): Path to the COCO JSON file.
        image_dir (str): Path to the directory containing images.
        output_csv_path (str): Path to save the output CSV file.
    """
    # Load COCO JSON
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # Initialize a dictionary to store person counts for each image
    person_class_id = 1  # COCO 'person' class ID
    image_person_count = {img['id']: 0 for img in coco_data['images']}

    # Count the number of people for each image
    for annotation in coco_data['annotations']:
        if annotation['category_id'] == person_class_id:
            image_id = annotation['image_id']
            image_person_count[image_id] += 1

    # Write the results to a CSV file
    with open(output_csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write header
        csv_writer.writerow(['image_id', 'image_name', 'person_count'])

        for image in coco_data['images']:
            image_id = image['id']
            image_name = image['file_name']
            person_count = image_person_count[image_id]
            csv_writer.writerow([image_id, image_name, person_count])

    print(f"CSV file saved to: {output_csv_path}")

# Example usage
if __name__ == "__main__":
    data_dir = "data/coco_dataset"
    create_people_count_csv(
        coco_json_path=os.path.join(data_dir, "annotations/instances_train2017.json"),
        image_dir=os.path.join(data_dir, "train2017"),
        output_csv_path=os.path.join(data_dir, "train_people_count.csv"),
    )
    create_people_count_csv(
        coco_json_path=os.path.join(data_dir, "annotations/instances_val2017.json"),
        image_dir=os.path.join(data_dir, "val2017"),
        output_csv_path=os.path.join(data_dir, "val_people_count.csv"),
    )

