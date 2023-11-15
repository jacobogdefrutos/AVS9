from segment_anything import SamPredictor, sam_model_registry
from utils import get_loaders
from model import ModelSimple
from train import train_one_epoch
import numpy as np
import torch.optim as optim
import torch
import os
DATA_IMG_DIR= '/home/jacobo15defrutos/AVS9/Data/Data_seg_SAM/train/Imag'
DATA_MASK_DIR= '/home/jacobo15defrutos/AVS9/Data/Data_seg_SAM/train/Labels'
BATCH_SIZE=1
NUM_WORKERS=1
PIN_MEMORY=True
NUM_EPOCHS=1
LEARNING_RATE = 0.001


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModelSimple()
    model.setup()
    transform = model.transform
    
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
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) #THE OPTIMIZER CAN BE CHANGED
    best_valid_loss = float('inf')
    for epoch in range(NUM_EPOCHS):
        running_vloss = 0.
        model.train(True)
        avg_batchloss = train_one_epoch(model, train_loader, optimizer, epoch,device)
    print("End of main")

if __name__ == "__main__":
    main()