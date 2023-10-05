import numpy as np
import torchvision
import torch.nn as nn
import cv2
import torch
import utils
import torch.optim as optim
from PIL import Image
from torchvision import transforms
from train import train_fn
from sklearn.metrics import f1_score,jaccard_score,recall_score,precision_score   
from deeplabv3 import createDeepLabv3, DeepLabV3WithSoftmax
#from torch.utils.tensorboard import SummaryWriter

# Hyperparameters etc.
LEARNING_RATE = 1e-3
# Settings for the image
BATCH_SIZE = 4
NUM_EPOCHS = 100
NUM_WORKERS = 8 
IMAGE_HEIGHT = 360 # 720 originally
IMAGE_WIDTH = 540   # 1080 originally
PIN_MEMORY = True
# Load/Save Settings
LOAD_MODEL = False
LOAD_CHECKPOINT = False
SAVE_CHECKPOINT = True
# Directories
#TRAIN_IMG_DIR = r"D:\Users\jacob\AVS9\Data_prueba\train\Imag" #LOCAL
#TRAIN_MASK_DIR = r"D:\Users\jacob\AVS9\Data_prueba\train\Labels"#LOCAL
#VAL_IMG_DIR = r"D:\Users\jacob\AVS9\Data_prueba\val\Imag"#LOCAL
#VAL_MASK_DIR =  r"D:\Users\jacob\AVS9\Data_prueba\val\Labels"#LOCAL
#CHECKPOINT_DIR = r"..\saved_models\my_checkpoint.pth.tar"#LOCAL

TRAIN_IMG_DIR = r"/home/jacobo15defrutos/AVS9_1/Data/train/Imag" #VM
TRAIN_MASK_DIR = r"/home/jacobo15defrutos/AVS9_1/Data/train/Labels"#VM
VAL_IMG_DIR = r"/home/jacobo15defrutos/AVS9_1/Data/val/Imag"#VM
VAL_MASK_DIR =  r"/home/jacobo15defrutos/AVS9_1/Data/val/Labels"#VM
CHECKPOINT_DIR = r"../saved_models/my_checkpoint.pth.tar"#VM


# define the torchvision image transforms
transform_imag = transforms.Compose([

    transforms.ToTensor(),
    transforms.Resize((IMAGE_HEIGHT,IMAGE_WIDTH),antialias=True),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
transform_mask = transforms.Compose([

    transforms.ToTensor(),
    transforms.Resize((IMAGE_HEIGHT,IMAGE_WIDTH),antialias=True),
    #transforms.Normalize(mean=[0.449], std=[0.226])
])

def main():
    print("Preparing to train data with the following settings")
    print("     Batch Size: ", BATCH_SIZE)
    print("     Number of Workers: ", NUM_WORKERS)
    print("     Number of Epochs: ", NUM_EPOCHS)
    print("     Image Size (wxh):", IMAGE_WIDTH, "x", IMAGE_HEIGHT)
    print("     Load Checkpoint: ", LOAD_CHECKPOINT)

    # Create instance for save best model class
    best_model = utils.save_best_model()

   # set computation device and training things
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #model = createDeepLabv3_softmax()
    model = DeepLabV3WithSoftmax(inputchannels=3, outputchannels=2)
    loss_fn= nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) #THE OPTIMIZER CAN BE CHANGED

    train_loader, val_loader = utils.get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        transform_imag,
        transform_mask,
        NUM_WORKERS,
        PIN_MEMORY,
    )
    # Setup tensorboard for visualizing the train loss and validation loss
    #writer = SummaryWriter("tensorboard/")
    metrics= {'f1_score': f1_score, 'PPV': precision_score, 'IoU': jaccard_score, 'Recall': recall_score }
    scaler = torch.cuda.amp.GradScaler()
    # Loop all epochs and train
    loaded_epoch = 0
    for epoch in range(loaded_epoch, NUM_EPOCHS):
        print("-------------------------------------")
        print("Epoch: ", epoch)
        # Train the model for this epoch and return the loss for that epoch
        mean_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler)
        #Add loss to tensorboard for visualization
        #writer.add_scalar('Loss/train', mean_loss, epoch)
        #writer.close()
        #Check accuracy of the model using the validation data
        current_val_loss = utils.check_accuracy(val_loader,metrics, model, epoch, loss_fn, device=DEVICE)
        #Save model if validation loss is best
        best_model(current_val_loss, epoch, model, optimizer, loss_fn)
       


    print('End of main')

if __name__ == "__main__":
    main()