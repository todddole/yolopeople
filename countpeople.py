from ultralytics import YOLO
import cv2
import os

# Initialize the YOLOv8 model
def initialize_model(model_path="yolov8n.pt"):
    """
    Initialize YOLOv8 model.
    Args:
        model_path (str): Path to YOLOv8 model. Use 'yolov8n.pt' for a pre-trained model.
    Returns:
        YOLO: The YOLOv8 model instance.
    """
    model = YOLO(model_path)
    return model

# Perform detection and count people
def count_people(image_path, model, class_id=0, confidence_threshold=0.5):
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

    return people_count, person_boxes, image

# Process a directory of images
def process_directory(image_dir, output_dir, model, class_id=0, confidence_threshold=0.5):
    """
    Process all images in a directory, count people, and save annotated images.
    Args:
        image_dir (str): Directory containing input images.
        output_dir (str): Directory to save annotated images.
        model (YOLO): YOLOv8 model instance.
        class_id (int): Class ID for 'person'.
        confidence_threshold (float): Confidence threshold for filtering detections.
    """
    os.makedirs(output_dir, exist_ok=True)

    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        if not image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue  # Skip non-image files

        # Count people in the image
        people_count, _, annotated_image = count_people(
            image_path, model, class_id, confidence_threshold
        )
        print(f"{image_name}: {people_count} people detected.")

        # Save annotated image
        output_path = os.path.join(output_dir, image_name)
        cv2.imwrite(output_path, annotated_image)

# Main function
if __name__ == "__main__":
    # Paths
    model_path = "yolov8l.pt"  # Replace with your trained model if needed
    image_dir = "data/test" #"data/coco_dataset/val/images"  # Directory containing test images
    output_dir = "output/annotated_images"  # Directory to save annotated images

    # Initialize model
    model = initialize_model(model_path)

    # Process images in the directory
    process_directory(image_dir, output_dir, model, class_id=0, confidence_threshold=0.5)