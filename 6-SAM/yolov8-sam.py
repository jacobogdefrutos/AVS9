from ultralytics import YOLO
from collections import defaultdict
from segment_anything.utils.transforms import ResizeLongestSide
import utils
from sklearn.metrics import f1_score,jaccard_score,recall_score,precision_score   
from utils import get_loaders, val_loss,find_number_in_string
from model import ModelSimple
from train import train_one_epoch
import numpy as np
import torch.optim as optim
import torch
import os
DATA_IMG_DIR= '/home/jacobo15defrutos/AVS9/Data/Data_seg_SAM/train/images'
DATA_MASK_DIR= '/home/jacobo15defrutos/AVS9/Data/Data_seg_SAM/train/labels'
BATCH_SIZE=1
NUM_WORKERS=8
PIN_MEMORY=True
NUM_EPOCHS=10
LEARNING_RATE = 0.001


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_yolo = YOLO("/home/jacobo15defrutos/AVS9/6-SAM/runs/detect/train/weights/best.pt")
    #model_yolo = model_yolo.load(weights='/home/jacobo15defrutos/AVS9/yolov8s.pt')
    model_sam = ModelSimple()
    model_sam.setup()
    transform = model_sam.transform
    #Use Yolo to get the boxes en each one of the train data images
    preds= model_yolo.predict(DATA_IMG_DIR)
    boxes_dic= defaultdict(dict)
    for pred in preds:
        file_name = pred.path
        number= find_number_in_string(str(file_name).split('/')[-1][:-4])
        #por ahora nos centramos solo en coger un box, la de mayor conf
        if len(pred.boxes.data)!=0:#There is a detection
            _,max_index = torch.max(pred.boxes.conf, dim=0)
            box= pred.boxes[max_index.item()]
            cords = box.xyxy[0].tolist()
            x_min, y_min, x_max, y_max = map(int, cords)
            box_sam=[x_min, y_min, x_max, y_max]
            boxes_dic[number]['cords']=np.array(box_sam)
        else:
            x_min,y_min,x_max,y_max =[0,0,pred.orig_shape[1],pred.orig_shape[0]]
            box_sam=[x_min, y_min, x_max, y_max]
            boxes_dic[number]['cords']=np.array(box_sam)
    
    #Load the data
    train_loader, val_loader = get_loaders(
        DATA_IMG_DIR,
        DATA_MASK_DIR,
        BATCH_SIZE,
        transform,
        NUM_WORKERS,
        PIN_MEMORY,
    )
    #Train the model
    best_model = utils.save_best_model()
    metrics= {'f1_score': f1_score, 'PPV': precision_score, 'IoU': jaccard_score, 'Recall': recall_score }
    optimizer = optim.Adam(model_sam.parameters(), lr=LEARNING_RATE) #THE OPTIMIZER CAN BE CHANGED
    best_valid_loss = float('inf')
    val_folder= '/home/jacobo15defrutos/AVS9/6-SAM/Results_val_seg'
    for epoch in range(NUM_EPOCHS):
        print(f"-----------Epoch: {epoch}------------")
        running_vloss = 0.
        model_sam.train(True)
        avg_batchloss = train_one_epoch(model_sam, train_loader, boxes_dic,transform,optimizer, epoch,device)
        avg_valloss = val_loss(val_loader,model_sam,boxes_dic,transform,metrics,epoch,val_folder,device)
        best_model(avg_valloss, epoch, model_sam, optimizer)

    print("End of main")

if __name__ == "__main__":
    main()