import os
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import json
import pickle

# Check if GPU is available
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Load the CSV file containing image names and counts
csv_path = "data/coco_dataset/train_people_count.csv"  # Replace with your CSV file path
image_folder = "data/coco_dataset/train/images"  # Replace with your image folder path
df = pd.read_csv(csv_path)

val_csv_path = "data/coco_dataset/val_people_count.csv"  # Replace with your CSV file path
val_image_folder = "data/coco_dataset/val/images"  # Replace with your image folder path
val_df = pd.read_csv(val_csv_path)

# Ensure the data contains the required columns
assert "image_name" in df.columns and "person_count" in df.columns, "CSV must contain 'image_name' and 'people_count' columns."
assert "image_name" in val_df.columns and "person_count" in val_df.columns, "CSV must contain 'image_name' and 'people_count' columns."


# Load images and their corresponding counts
def load_data(image_folder, df, img_size):
    images = []
    labels = []
    for _, row in df.iterrows():
        img_path = os.path.join(image_folder, row["image_name"])
        if os.path.exists(img_path):
            img = load_img(img_path, target_size=(img_size, img_size))  # Resize image
            img = img_to_array(img) / 255.0  # Normalize to [0, 1]
            images.append(img)
            labels.append(row["person_count"])
    return np.array(images), np.array(labels)

# Define image size
IMG_SIZE = 512  # Resize all images to 512x512
BATCH_SIZE = 32

# Define the regression model
def build_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='linear'))  # Single output for people count
    return model

# Build and compile the model
model = build_model((IMG_SIZE, IMG_SIZE, 3))
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Create a data generator
datagen = ImageDataGenerator(
    rescale=1.0/255,  # Normalize pixel values to [0, 1]
    horizontal_flip=True,
    rotation_range=10,
    zoom_range=0.2
)

val_datagen = ImageDataGenerator(
    rescale=1.0/255)

# Create training and validation generators
train_generator = datagen.flow_from_dataframe(
    dataframe=df,
    directory=image_folder,
    x_col="image_name",
    y_col="person_count",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="raw",  # Raw for regression tasks
    shuffle=True
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=val_image_folder,
    x_col="image_name",
    y_col="person_count",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="raw",
    shuffle=False
)

# Fit the model

EPOCHS = 10

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    verbose=1
)

#Evaluate the model
# Predict on the validation set
val_predictions = model.predict(val_generator)

# Extract true labels from the validation generator
true_labels = val_generator.labels  # Extract the true labels

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(true_labels, val_predictions)
print(f"Validation MAE: {mae}")

# Save the model
model.save("people_counting_model.h5")

# Save loss plot
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('training_loss_plot.png')
plt.close()

# Save MAE plot
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.savefig('training_mae_plot.png')
plt.close()

# Save training history as JSON
with open('training_history.json', 'w') as f:
    json.dump(history.history, f)

# Save training history as Pickle
with open('training_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)