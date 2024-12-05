from ultralytics import YOLO
import os

# Define paths and parameters
def main():
    # Paths to dataset and output directory
    data_yaml = "data/person_data.yaml"  # YAML file defining the dataset
    model_path = "yolov8l.pt"  # Pre-trained YOLOv8 model
    output_dir = "runs/train"  # Directory to save training results
    epochs = 100  # Number of training epochs
    img_size = 640  # Input image size
    batch_size = 16  # Batch size

    # Train YOLOv8 model
    print(f"Starting training with model: {model_path}")
    model = YOLO(model_path)  # Load pre-trained YOLOv8 model
    model.train(
        data=data_yaml,  # Path to dataset YAML
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        project=output_dir,
        name="yolov8l_people",  # Run name
        workers=4,  # Number of data loader workers
        device=0,# Use GPU 0 for training
        amp=True
    )
    print("Training complete.")

# Run the script
if __name__ == "__main__":
    main()
