from pathlib import Path
from typing import Any, Callable, Optional
import os
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms
from torchvision.datasets.vision import VisionDataset

IMAGE_HEIGHT = 480 # 720 originally
IMAGE_WIDTH = 720   # 1080 originally
TRAIN_IMG_DIR = r"D:\Users\jacob\Project\Data\train\Imag"
TRAIN_MASK_DIR = r"D:\Users\jacob\Project\Data\train\Labels"



class IrisDataset(VisionDataset):
    """A PyTorch dataset for image segmentation task.
    The dataset is compatible with torchvision transforms.
    The transforms passed would be applied to both the Images and Masks.
    """
    def __init__(self,image_dir: str,mask_dir: str,transform_imag: Optional[Callable] = None, transform_mask: Optional[Callable] = None,) -> None:
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir) 
        self.transform_imag = transform_imag
        self.transform_mask = transform_mask
        self.images = sorted(self.image_dir.glob("*"))
        self.masks = sorted(self.mask_dir.glob("*"))
        #indices_i = np.arange(len(images))
        #indices_m = np.arange(len(masks))
        #self.image_list = images[indices_i]
        #self.mask_list = masks[indices_m]
    def __len__(self) -> int:
        return len(self.images)
    def __getitem__(self, index: int) -> Any:
        img_path = self.images[index]
        mask_path = self.masks[index]
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("RGB"),dtype=np.uint8)
        mask= mask[:,:,0]
        img_name = str(img_path).split('/')[-1][:-4]
        mask_name = str(mask_path).split('/')[-1][:-4]
        sample = {"image": image, "mask": mask, "image_name":img_name,"mask_name": mask_name}
        if self.transform_imag and self.transform_mask:
            sample["image"] = self.transform_imag(sample["image"])
            sample["mask"] = self.transform_mask(sample["mask"])
        return sample

