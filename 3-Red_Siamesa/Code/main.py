import numpy as np
import torch
import torch.nn.functional as F #de aqui podemos coger la funcion pairwise_distance para calcula la ecludian_distance
from utils import get_loaders, ContrastiveLoss, train_fn, val_loss
from torchvision import transforms
import torch.optim as optim
from lenet5 import LeNet5
import matplotlib.pyplot as plt

TRAIN_DIR= r'Data\saved_test_clas_images\train'
VAL_DIR= r'Data\saved_test_clas_images\val'
train_csv='Iris_train_seg_list.csv'
val_csv='Iris_val_seg_list.csv'
BATCH_SIZE= 2
NUM_WORKERS=4
NUM_EPOCHS=20
IMAGE_HEIGHT=540
IMAGE_WIDTH=810
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    # define the torchvision image transforms
    
    transform = transforms.Compose([

        transforms.ToTensor(),
        transforms.Resize((IMAGE_HEIGHT,IMAGE_WIDTH),antialias=True),
        transforms.Normalize(mean=[0.0], std=[1.0])
    ])

    train_loader, val_loader= get_loaders(train_csv,val_csv,TRAIN_DIR,VAL_DIR,BATCH_SIZE,transform,NUM_WORKERS)
    loss_fn = ContrastiveLoss()
    model= LeNet5().to(device=DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(0, NUM_EPOCHS):
        print("-------------------------------------")
        print("Epoch: ", epoch)
        train_loss,train_losses_list, counter= train_fn(train_loader,model,optimizer,loss_fn,device=DEVICE)
        vall_loss,val_losses_list, counter= val_loss(val_loader,model,optimizer,loss_fn,device=DEVICE)
    fig, axs = plt.subplots(1, 2, figsize=(9, 3), sharey=True)
    axs[0].plot(counter,train_losses_list)
    axs[1].plot(counter,val_losses_list)

    print("End of main")
if __name__ == "__main__":
    main()    