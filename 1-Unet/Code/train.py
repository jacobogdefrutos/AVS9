from utils import get_loaders, get_test
import torch
import os
import time
from tqdm import tqdm
import  numpy as np






def train_fn(loader, model, optimizer, loss_fn,scaler):
    #since = time.time()
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(DEVICE)
    losses = []
    model.train()
    # Data size is [batches, in_channels, image_height, image_width]
    for sample in tqdm(iter(loader)):
        data = sample['image'].to(device=DEVICE)
        targets = sample['mask'].long().to(device=DEVICE)
        targets=targets.squeeze(1)
        optimizer.zero_grad() # zero the parameter gradients
        #forward
        with torch.set_grad_enabled(True):
            predictions = model(data)
            #new 
            outputs=predictions
            #outputs = predictions['out'].data comento esto para el runeo de UNET
            loss = loss_fn(outputs, targets).requires_grad_()
            #print("Loss: ", loss.item())
            #backward
            
            loss.backward()
            optimizer.step()
        #scaler.update()

        #update tqdm loop
        #loop.set_postfix(loss=loss.item())
        losses.append(loss.item())
    mean_loss = np.mean(losses)
    #time_elapsed = time.time() - since
    #print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print("Training Loss: ", mean_loss)
    return mean_loss
