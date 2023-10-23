import os
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
        img0 = img0.convert("L")
        img1 = img1.convert("L")
        # Apply image transformations
        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        return img0, img1 , label
    def __len__(self):
        return len(self.df)