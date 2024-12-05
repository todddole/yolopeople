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
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError


log_dir = "logs/fit" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard_callback = TensorBoard(log_dir = log_dir, histogram_freq=1)

checkpoint = ModelCheckpoint(
    "checkpoint_epoch{epoch:02d}.keras",
    save_best_only=False,
    save_weights_only=False,
    verbose=1
    )

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10, # Stop if no improvement for 10 epochs
    restore_best_weights=True,
    verbose=1
    )



# Check if GPU is available
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth set.")
    except RuntimeError as e:
        print(e)

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
IMG_SIZE = 224  # Resize all images to 512x512
BATCH_SIZE = 32

# Define the regression model
def build_model(input_shape):

    # Load VGG16 with custom input size
    vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    # Add custom layers for regression
    model = Sequential()
    model.add(vgg16)
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))  # Fully connected layer
    model.add(Dense(1, activation='linear'))   # Single output for regression

    sgd = SGD(learning_rate=0.0001, momentum=0.9)

    # Compile the model
    model.compile(optimizer=sgd, loss='mse', metrics=['mae'])

    print(model.summary())

    return model

# Build and compile the model
model_path = "models/vgg_people_counting_model.h5"
custom_objects = {
    'mse': MeanSquaredError()
}
model = load_model(model_path, custom_objects=custom_objects)
for layer in model.layers[-4:]:
    layer.trainable = True
sgd = SGD(learning_rate=0.0001, momentum=0.9)

# Compile the model
model.compile(optimizer=sgd, loss='mse', metrics=['mae'])

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

EPOCHS = 72
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)

history = model.fit(
    train_generator,
    validation_data=val_generator,
    initial_epoch=39,
    epochs=EPOCHS,
    callbacks = [tensorboard_callback, lr_scheduler, checkpoint, early_stopping],
    verbose=1
)

#Evaluate the model
# Predict on the validation set
val_predictions = model.predict(val_generator)

# Extract true labels from the validation generator
true_labels = val_df["person_count"].values

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(true_labels, val_predictions)
print(f"Validation MAE: {mae}")

# Save the model
model.save("vgg_finetuned_people_counting_model.h5")

# Save loss plot
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('vgg_training_loss_plot.png')
plt.close()

# Save MAE plot
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.savefig('vgg_training_mae_plot.png')
plt.close()

# Save training history as JSON
with open('vgg_training_history.json', 'w') as f:
    json.dump(history.history, f)

# Save training history as Pickle
with open('vgg_training_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)

np.save('val_predictions.npy', val_predictions)
np.save('true_labels.npy', true_labels)
