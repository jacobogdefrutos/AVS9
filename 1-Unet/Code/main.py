import numpy as np
import torchvision
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt
import torch
import utils
import torch.optim as optim
from PIL import Image
from torchvision import transforms
from train import train_fn
from sklearn.metrics import f1_score,jaccard_score,recall_score,precision_score   
import segmentation_models_pytorch as smp
#from torch.utils.tensorboard import SummaryWriter

# Hyperparameters etc.
LEARNING_RATE = 0.001
# Settings for the image
BATCH_SIZE = 5
NUM_EPOCHS = 20
NUM_WORKERS = 8
IMAGE_HEIGHT = 704 # 720 originally
IMAGE_WIDTH = 704   # 1080 originally
PIN_MEMORY = True
# UNet Model transfer learning 
ACTIVATION = "softmax2d"
ENCODER_NAME = "timm-gernet_l"
ENCODER_WEIGHTS= "imagenet"
# Load/Save Settings
LOAD_MODEL = False
LOAD_CHECKPOINT = False
SAVE_CHECKPOINT = True
# Directories
TRAIN_IMG_DIR = r"/home/jacobo15defrutos/AVS9/Data/Data_new_SAM/train/images"
TRAIN_MASK_DIR = r"/home/jacobo15defrutos/AVS9/Data/Data_new_SAM/train/labels"
VAL_IMG_DIR = r"/home/jacobo15defrutos/AVS9/Data/Data_new_SAM/val/images"
VAL_MASK_DIR =  r"/home/jacobo15defrutos/AVS9/Data/Data_new_SAM/val/labels"


# define the torchvision image transforms
transform_imag = transforms.Compose([

    transforms.ToTensor(),
    transforms.Resize((IMAGE_HEIGHT,IMAGE_WIDTH),antialias=True),
    transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
])
transform_mask = transforms.Compose([

    transforms.ToTensor(),
    transforms.Resize((IMAGE_HEIGHT,IMAGE_WIDTH),antialias=True)
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
    #model = createDeepLabv3()
        # Create the UNet model
    # Transfer learning model
    model = smp.Unet(
        encoder_name=ENCODER_NAME,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=3,
        classes=2,
        activation=ACTIVATION,
    ).to(DEVICE)

    # No transfer learning model
    #model = UNET(in_channels=3, out_channels=6).to(DEVICE)
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
    train_losses = []
    val_losses = []
    for epoch in range(loaded_epoch, NUM_EPOCHS):
        print("-------------------------------------")
        print("Epoch: ", epoch)
        # Train the model for this epoch and return the loss for that epoch
        mean_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler)
        train_losses.append(mean_loss)
        #Add loss to tensorboard for visualization
        #writer.add_scalar('Loss/train', mean_loss, epoch)
        #writer.close()
        #Check accuracy of the model using the validation data
        folder_val =r"/home/jacobo15defrutos/AVS9/Data/saved_val_images_unet"
        current_val_loss = utils.check_accuracy(val_loader,metrics, model, epoch, loss_fn,folder_val, device=DEVICE)
        val_losses.append(current_val_loss)
        #Save model if validation loss is best
        best_model(current_val_loss, epoch, model, optimizer, loss_fn)
       
    epochs = range(1, NUM_EPOCHS + 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.show()

    print('End of main')

if __name__ == "__main__":
    main()