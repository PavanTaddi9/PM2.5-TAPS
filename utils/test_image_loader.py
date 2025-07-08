import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
class ImageDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.image_files = [f for f in os.listdir(directory) if f.endswith(('.jpeg', '.jpg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.directory, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")  # Ensure the image is RGB
        
        if self.transform:
            transforms.Resize((224, 224))
            image = self.transform(image)
        
        return image

def check_loader():
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # Resize to a fixed size
        transforms.ToTensor(),         # Convert image to tensor and normalize to [0, 1] by dividing by 255
    ])
    
    # Initialize the dataset
    image_directory = '/kaggle/input/202444'
    dataset = ImageDataset(directory=image_directory, transform=transform)
    
    # Create a DataLoader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    return dataloader

def predict(model, device, epoch, data_loader):
    y_pred = []  # Initialize an empty list to store predictions
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():  # Disable gradient computation for inference
        for batch in data_loader:
            batch = batch.to(device)  # Move batch to the specified device
            predicted = model(batch, epoch)  # Forward pass
            y_pred.append(predicted)  # Append predictions to the list
    
    y_pred = torch.cat(y_pred, dim=0)  # Concatenate all tensors along the batch dimension
    return y_pred
    