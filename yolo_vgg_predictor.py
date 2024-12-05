import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.losses import MeanSquaredError
import logging

logging.getLogger('ultralytics').setLevel(logging.WARNING)

def load_and_prepare_image(image_path, img_size=224):
    """Load an image file and prepare it for prediction."""
    img = load_img(image_path, target_size=(img_size, img_size))
    img = img_to_array(img) / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img


def predict_counts(df, model_path, column_name):
    """Predict people counts using the specified model and update the DataFrame."""
    custom_objects = {
        'mse': MeanSquaredError()
    }
    model = load_model(model_path, custom_objects=custom_objects)
    img_size = model.input_shape[1]  # Assuming width and height are the same

    predictions = []
    for _, row in df.iterrows():
        img_path = f"data/coco_dataset/val/images/{row['image_name']}"
        img = load_and_prepare_image(img_path, img_size)
        pred_count = model.predict(img).flatten()[0]  # Get the scalar prediction
        predictions.append(pred_count)

    df[column_name] = predictions


def main():
    # Load existing CSV file
    df = pd.read_csv("yolo_val_predictions.csv")

    # Define models and the columns to add
    models_info = {
        "models/vgg_people_counting_model.h5": "vgg16_frozen_pred",
        "vgg_finetuned_people_counting_model.h5": "vgg16_finetuned_pred"
    }

    # Run predictions for each model and update the DataFrame
    for model_path, column_name in models_info.items():
        predict_counts(df, model_path, column_name)

    # Save the updated DataFrame
    df.to_csv("updated_val_predictions.csv", index=False)
    print("Updated predictions saved to 'updated_val_predictions.csv'.")


if __name__ == "__main__":
    main()
