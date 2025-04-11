# coding: utf-8

# # Demo for running IFCNN to fuse multiple types of images

import os
import torch
from model import myIFCNN

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from utils.myDatasets import MRAT1Dataset

# ## 2. Load the well-trained image fusion model (IFCNN-MAX)

fuse_scheme = 0  # Use IFCNN-MAX for fusion
if fuse_scheme == 0:
    model_name = 'IFCNN-MAX'
elif fuse_scheme == 1:
    model_name = 'IFCNN-SUM'
elif fuse_scheme == 2:
    model_name = 'IFCNN-MEAN'
else:
    model_name = 'IFCNN-MAX'

# Load pretrained model
model = myIFCNN(fuse_scheme=fuse_scheme)
model.load_state_dict(torch.load('snapshots/' + model_name + '.pth', weights_only=True))
model.eval()
model = model.cuda()

# ## 3. New Fusion Part: Fuse MRA and T1 Images

# Paths to the T1 and MRA folders
t1_folder = '/data/student_1/XDataset_backup/IXI-T1'
mra_folder = '/data/student_1/XDataset_backup/IXI-MRA'

# Create dataset and dataloader
target_shape = (256, 256, 150)  # Ensure all images are resized or cropped to this shape
transform = transforms.Compose([transforms.Normalize(mean=[0.5], std=[0.5])])
dataset = MRAT1Dataset(t1_folder, mra_folder, transform=transform, target_shape=target_shape)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for t1, mra in dataloader:
        t1, mra = t1.cuda(), mra.cuda()

        # Forward pass
        fused = model(t1, mra)

        # Compute loss
        loss = criterion(fused, mra)  # Example: Use MRA as the target
        epoch_loss += loss.item()

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(dataloader):.4f}")

# Save the trained model
torch.save(model.state_dict(), 'snapshots/IFCNN-MAX-MRA-T1.pth')

print("Fused image shape:", fused.shape)
print(f"Padding tensor with shape: {tensor.shape}")
