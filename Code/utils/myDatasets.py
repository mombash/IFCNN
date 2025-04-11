# coding: utf-8
import os
import torch
import torch.utils.data as data
from torchvision import datasets, transforms
from PIL import ImageChops
import os
import nibabel as nib
import torch
from torch.utils.data import Dataset
import numpy as np

class MRAT1Dataset(Dataset):
    def __init__(self, t1_folder, mra_folder, transform=None, target_shape=(256, 256, 150)):
        self.t1_files = sorted([os.path.join(t1_folder, f) for f in os.listdir(t1_folder) if f.endswith('.nii.gz')])
        self.mra_files = sorted([os.path.join(mra_folder, f) for f in os.listdir(mra_folder) if f.endswith('.nii.gz')])
        self.transform = transform
        self.target_shape = target_shape  # Target shape for all images

    def __len__(self):
        return len(self.t1_files)

    def __getitem__(self, idx):
        t1_path = self.t1_files[idx]
        mra_path = self.mra_files[idx]

        # Load NIfTI files
        t1_img = nib.load(t1_path).get_fdata()
        mra_img = nib.load(mra_path).get_fdata()

        # Normalize to [0, 1]
        t1_img = (t1_img - t1_img.min()) / (t1_img.max() - t1_img.min() + 1e-8)
        mra_img = (mra_img - mra_img.min()) / (mra_img.max() - mra_img.min() + 1e-8)

        # Resize or crop to target shape
        t1_img = self.resize_or_crop(t1_img, self.target_shape)
        mra_img = self.resize_or_crop(mra_img, self.target_shape)

        # Convert to tensors
        t1_tensor = torch.from_numpy(t1_img).float().unsqueeze(0)  # Add channel dimension
        mra_tensor = torch.from_numpy(mra_img).float().unsqueeze(0)  # Add channel dimension

        if self.transform:
            t1_tensor = self.transform(t1_tensor)
            mra_tensor = self.transform(mra_tensor)

        return t1_tensor, mra_tensor

    def resize_or_crop(self, img, target_shape):
        """Resize or crop the image to the target shape."""
        current_shape = img.shape
        # Crop if the image is larger than the target shape
        slices = [slice(0, min(current, target)) for current, target in zip(current_shape, target_shape)]
        img = img[tuple(slices)]

        # Pad if the image is smaller than the target shape
        pad_width = [(0, max(0, target - current)) for current, target in zip(current_shape, target_shape)]
        img = np.pad(img, pad_width, mode='constant', constant_values=0)

        return img
    
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif'
]

class ImagePair(data.Dataset):
    def __init__(self, impath1, impath2, mode='RGB', transform=None):
        self.impath1 = impath1
        self.impath2 = impath2
        self.mode = mode
        self.transform = transform
        
    def loader(self, path):
        return Image.open(path).convert(self.mode)
    
    def is_image_file(self, filename):
        return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

    def get_pair(self):
        if self.is_image_file(self.impath1):
            img1 = self.loader(self.impath1)
        if self.is_image_file(self.impath2):
            img2 = self.loader(self.impath2)
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2

    def get_source(self):
        if self.is_image_file(self.impath1):
            img1 = self.loader(self.impath1)
        if self.is_image_file(self.impath2):
            img2 = self.loader(self.impath2)
        return img1, img2

class ImageSequence(data.Dataset):
    def __init__(self, is_folder=False, mode='RGB', transform=None, *impaths):
        self.is_folder = is_folder
        self.mode = mode
        self.transform = transform
        self.impaths = impaths

    def loader(self, path):
        return Image.open(path).convert(self.mode)

    def get_imseq(self):
        if self.is_folder:
            folder_path = self.impaths[0]
            impaths = self.make_dataset(folder_path)
        else:
            impaths = self.impaths

        imseq = []
        for impath in impaths:
            if os.path.exists(impath):
                im = self.loader(impath)
                if self.transform is not None:
                    im = self.transform(im)
                imseq.append(im)
        return imseq

    def is_image_file(self, filename):
        return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

    def make_dataset(self, img_root):
        images = []
        for root, _, fnames in sorted(os.walk(img_root)):
            for fname in fnames:
                if self.is_image_file(fname):
                    img_path = os.path.join(img_root, fname)
                    images.append(img_path)
        return images