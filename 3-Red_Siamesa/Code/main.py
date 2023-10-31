import numpy as np
import torch
import torch.nn.functional as F #de aqui podemos coger la funcion pairwise_distance para calcula la ecludian_distance
from utils import get_loaders, ContrastiveLoss, train_fn, val_loss,save_best_model
from torchvision import transforms
import torch.optim as optim
from lenet5 import LeNet5
import matplotlib.pyplot as plt

TRAIN_DIR= r'Data/saved_seg_class_images/train_siamesa'
VAL_DIR= r'Data/saved_seg_class_images/val'
train_csv='Iris_train_siamesa_seg_list.csv'
val_csv='Iris_val_seg_list.csv'
BATCH_SIZE= 10
NUM_WORKERS=8
NUM_EPOCHS=20
IMAGE_HEIGHT=224
IMAGE_WIDTH=224
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
    best_model = save_best_model()
    model= LeNet5().to(device=DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    counter=[]
    trainl_list= []
    vall_list=[]
    for epoch in range(0, NUM_EPOCHS):
        counter.append(epoch)
        print("-------------------------------------")
        print("Epoch: ", epoch)
        train_loss= train_fn(train_loader,model,optimizer,loss_fn,device=DEVICE)
        vall_loss= val_loss(val_loader,model,optimizer,loss_fn,device=DEVICE)
        trainl_list.append(train_loss)
        vall_list.append(vall_loss)
        #best_model(vall_loss, epoch, model, optimizer, loss_fn)


    fig, ax = plt.subplots()
    ax.plot(trainl_list,counter, label='Train Loss', color='blue')
    ax.plot(vall_list,counter, label='Validation Loss', color='red')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    plt.show()
    print("End of main")
if __name__ == "__main__":
    main()    