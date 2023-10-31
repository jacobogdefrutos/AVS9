import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
class SiameseDataset():
    def __init__(self,file_csv=None,files_dir=None,transform=None):
        # used to prepare the labels and images path
        self.df=pd.read_csv(file_csv)
        self.df.columns =["OI","OD","class_label","label_oi","label_od"]
        self.files_dir = files_dir   
        self.transform = transform

    def __getitem__(self,index):
        # getting the image path
        image1_path=os.path.join(self.files_dir,self.df.iat[index,0])
        image2_path=os.path.join(self.files_dir,self.df.iat[index,1])
        label=self.df.iat[index,2]#Cogemos la label que solo diferencia entre iris CMV o SURV
        # Loading the image
        img0 = Image.open(image1_path)
        img1 = Image.open(image2_path)
        img0 = np.array(img0.convert("L"))
        img1 = np.array(img1.convert("L"))
        img0_fft= np.fft.fftshift(np.fft.fft2(img0))
        img1_fft= np.fft.fftshift(np.fft.fft2(img1))
        img0_abs=np.log(abs(img0_fft))
        img1_abs=np.log(abs(img1_fft))
        img0_n=cv2.normalize(img0_abs,None,0, 255, cv2.NORM_MINMAX)
        img1_n=cv2.normalize(img1_abs,None,0, 255, cv2.NORM_MINMAX)
        img0_u8= np.uint8(img0_n)
        img1_u8= np.uint8(img1_n)
        # Apply image transformations
        if self.transform is not None:
            img0 = self.transform(img0_u8)
            img1 = self.transform(img1_u8)
        return img0, img1 , label
    def __len__(self):
        return len(self.df)