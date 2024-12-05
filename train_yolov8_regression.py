import os
import torch
from ultralytics import YOLO
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import pandas as pd

class PeopleCountingDataset(Dataset):
    def __init__(self, csv_file, image_folder, transform=None):
        """
        Args:
            csv_file (str): Path to the CSV file with image filenames and counts.
            image_folder (str): Path to the folder containing images.
            transform (callable, optional): Optional transforms to be applied on a sample.
        """
        self.data = pd.read_csv(csv_file)
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.image_folder, self.data.iloc[idx, 1])  # Assuming 2nd column is file name
        image = Image.open(img_name).convert("RGB")
        count = self.data.iloc[idx, 2]  # Assuming 3rd column is the count

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(count, dtype=torch.float32)


# Define any necessary image transformations
transform = transforms.Compose([
    transforms.Resize((640, 640)),  # Resize to YOLOv8 input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = PeopleCountingDataset(csv_file="data/coco_dataset/train_people_count.csv", image_folder="data/coco_dataset/train", transform=transform)
val_dataset = PeopleCountingDataset(csv_file="data/coco_dataset/val_people_count.csv", image_folder="data/coco_dataset/val", transform=transform)

# Print the number of items in each dataset
print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(val_dataset)}")

# Load pretrained YOLOv8 model
# Load YOLOv8 model
model = YOLO("yolov8n.pt").model

model.model[-1] = torch.nn.Linear(in_features=model.model[-1].in_features, out_features=1)
model.loss = torch.nn.MSELoss()



# Test the model with dummy input
dummy_input = torch.randn(1, 3, 640, 640)  # Batch size of 1, image size 640x640
output = model(dummy_input)
print("Regression output:", output)
exit(0)

# Prepare data
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Define optimizer and loss function
criterion = nn.MSELoss()
optimizer = optim.Adam(regression_model.parameters(), lr=1e-4)

num_epochs = 10
# Training loop
for epoch in range(num_epochs):
    regression_model.train()
    for images, labels in train_loader:
        outputs = regression_model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation step
    regression_model.eval()
    with torch.no_grad():
        val_loss = 0
        for images, labels in val_loader:
            outputs = regression_model(images)
            val_loss += criterion(outputs, labels).item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss / len(val_loader)}")

# Save the model
torch.save(regression_model.state_dict(), "regression_model.pth")