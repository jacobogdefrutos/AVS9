from pathlib import Path
import os
import cv2
import re
import torch
import numpy as np
from torch.utils.data import DataLoader
from segment_anything.utils.transforms import ResizeLongestSide
from PIL import Image
from torchvision import transforms
from torchvision.datasets.vision import VisionDataset

IMAGE_HEIGHT = 480 # 720 originally
IMAGE_WIDTH = 720   # 1080 originally
TRAIN_IMG_DIR = r"/home/jacobo15defrutos/AVS9/Data/Data_seg_SAM/train/Imag"
TRAIN_MASK_DIR = r"/home/jacobo15defrutos/AVS9/Data/Data_seg_SAM/train/Labels"
def find_number_in_string(input_string):
    # Regular expression to find a number in a string
    pattern = r'\d+'

    # Search for the pattern in the input string
    match = re.search(pattern, input_string)

    # Check if a match is found
    if match:
        # Extract and return the matched number
        return int(match.group())

    # Return None if no number is found
    return None


class IrisDataset(VisionDataset):
    """A PyTorch dataset for image segmentation task.
    The dataset is compatible with torchvision transforms.
    The transforms passed would be applied to both the Images and Masks.
    """
    def __init__(self,image_dir,mask_dir,transform=None) -> None:
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir) 
        self.transform=transform
        self.images = sorted(self.image_dir.glob("*"))
        self.masks = sorted(self.mask_dir.glob("*"))

    def __len__(self) -> int:
        return len(self.images)
    def __getitem__(self, index: int) :
        img_path = self.images[index]
        mask_path = self.masks[index]
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("RGB"))/255
        mask= mask.astype(np.uint8)
        image = cv2.resize(image, (400, 400))
        mask = cv2.resize(mask, (400, 400))
        mask= mask[:,:,1]#.astype(float)
        input_image = self.transform.apply_image(image)
        input_mask = self.transform.apply_image(mask)
        input_image_torch = torch.as_tensor(input_image,dtype=torch.float32)
        input_mask_torch = torch.as_tensor(input_mask,dtype=torch.uint8)
        input_mask_torch= input_mask_torch.unsqueeze(2)
        transformed_image = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
        transformed_mask = input_mask_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
        number= find_number_in_string( str(img_path).split('/')[-1][:-4])
        sample = {"image": transformed_image, "mask": transformed_mask, "idx":number}
        
        return sample

