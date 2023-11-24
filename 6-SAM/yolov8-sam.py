from ultralytics import YOLO
from collections import defaultdict
from resize import resize_image_with_pading
import matplotlib.pyplot as plt
import utils 
from utils import get_loaders, val_loss,find_number_in_string
from model import ModelSimple
from train import train_one_epoch
import numpy as np
import torch.optim as optim
import torch
import os
DATA_IMG_DIR= '/home/jacobo15defrutos/AVS9/Data/Data_new_SAM/train/images'
DATA_MASK_DIR= '/home/jacobo15defrutos/AVS9/Data/Data_new_SAM/train/labels'
BATCH_SIZE=1
NUM_WORKERS=8
PIN_MEMORY=True
NUM_EPOCHS=20
LEARNING_RATE = 0.001
NEW_SIZE = (800,800)



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_yolo = YOLO("/home/jacobo15defrutos/AVS9/6-SAM/saved_best_model/best.pt")
    model_sam = ModelSimple()
    model_sam.setup()
    transform = model_sam.transform
    #Use Yolo to get the boxes en each one of the train data images
    #resize_image_with_pading(DATA_IMG_DIR,DATA_IMG_DIR,NEW_SIZE)
    #resize_image_with_pading(DATA_MASK_DIR,DATA_MASK_DIR,NEW_SIZE)
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
    train_losses = []
    val_losses = []
    best_model = utils.save_best_model()
    optimizer = optim.Adam(model_sam.parameters(), lr=LEARNING_RATE) #THE OPTIMIZER CAN BE CHANGED
    best_valid_loss = float('inf')
    val_folder= '/home/jacobo15defrutos/AVS9/6-SAM/Results_val_seg'
    for epoch in range(NUM_EPOCHS):
        print(f"-----------Epoch: {epoch}------------")
        running_vloss = 0.
        model_sam.train(True)
        avg_batchloss = train_one_epoch(model_sam, train_loader,transform,boxes_dic,optimizer, epoch,device)
        avg_valloss = val_loss(val_loader,model_sam,transform,boxes_dic,epoch,val_folder,device)
        train_losses.append(avg_batchloss)
        val_losses.append(avg_valloss)
        best_model(avg_valloss, epoch, model_sam, optimizer)
    

    # Plotting
    epochs = range(1, NUM_EPOCHS + 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.show()

    print("End of main")

if __name__ == "__main__":
    main()