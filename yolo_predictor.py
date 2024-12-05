from ultralytics import YOLO
import os
import pandas as pd
import cv2
import os
import logging

logging.getLogger('ultralytics').setLevel(logging.WARNING)

def count_people(image_path, model, class_id=0, confidence_threshold=0.3):
    """
    Count the number of people in an image using YOLOv8.
    Args:
        image_path (str): Path to the input image.
        model (YOLO): YOLOv8 model instance.
        class_id (int): Class ID for 'person' (usually 0 in COCO).
        confidence_threshold (float): Confidence threshold for filtering detections.
    Returns:
        int: Number of people detected.
        list: Bounding boxes for detected people.
        ndarray: Annotated image.
    """
    # Run inference
    results = model(image_path)
    detections = results[0]

    # Filter for 'person' detections
    people_count = 0
    person_boxes = []
    for box in detections.boxes:
        if box.cls == class_id and box.conf >= confidence_threshold:
            people_count += 1
            person_boxes.append(box.xyxy[0])  # xyxy format bounding box

    # Load and annotate the image
    image = cv2.imread(image_path)
    for box in person_boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw bounding box

    return people_count

def predict():
    models = [
        ('yolov8n', 'yolov8n.pt'),
        ('yolov8l', 'yolov8l.pt'),
        ('yolov8n_people', 'runs/train/yolov8_people10/weights/best.pt'),
        ('yolov8l_people', "runs/train/yolov8l_people/weights/best.pt")
    ]
    csv_path = "data/coco_dataset/val_people_count.csv"  # CSV with image names and ground truth counts
    image_folder = "data/coco_dataset/val/images"  # Folder containing validation images
    df = pd.read_csv(csv_path)

    for model_name, model_path in models:
        # Paths to dataset and trained model
        print('predicting for '+model_name)

        # Load the trained YOLO model
        print(f"Loading model from: {model_path}")
        model = YOLO(model_path)  # Load trained YOLOv8 model

        # Load the CSV to get image names

        column_name = model_name + '_pred'
        # Prepare to collect predictions
        df[column_name] = 0
        print(df.columns)

        # Iterate through each image in the validation set
        for index, row in df.iterrows():
            image_path = os.path.join(image_folder, row['image_name'])
            # Predict using YOLO model

            # Count number of person detections
            person_count = count_people(image_path, model)

            # Save the results
            df.at[index, column_name] = person_count

    # Save to csv

    df.to_csv('yolo_val_predictions.csv', index=False)
    print("Predictions saved to 'yolo_val_predictions.csv'.")

# Run the prediction function
if __name__ == "__main__":
    predict()