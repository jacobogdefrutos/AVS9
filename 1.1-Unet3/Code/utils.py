import torch
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torchvision

import torchvision.transforms as transforms
from dataset_2 import IrisDataset
from torch.utils.data import DataLoader, random_split
from metrics import  calc_IoU, calc_sensitivity ,calc_PPV



#Esta funcion no creo que se vaya a utilizar
def draw_segmentation_map(output,mask_name,folder):
    epoch=400
    seg_map = output[0] # use only one output from the batch
    seg_map = torch.argmax(seg_map.squeeze(), dim=0).detach().cpu().numpy()
    label_color_map = np.array([[0.0, 0.0, 0.0], [255.0, 0.0, 0.0]], dtype=np.float32)
    red_map = np.zeros_like(seg_map).astype(np.uint8)
    green_map = np.zeros_like(seg_map).astype(np.uint8)
    blue_map = np.zeros_like(seg_map).astype(np.uint8)
    
    for label_num in range(0, len(label_color_map)):
        index = seg_map == label_num
        red_map[index] = np.array(label_color_map)[label_num][0]
        green_map[index] = np.array(label_color_map)[label_num][1]
        blue_map[index] = np.array(label_color_map)[label_num][2]
        
    rgb = np.stack([red_map, green_map, blue_map], axis=2)
    segmentation_map = np.array(rgb, dtype=np.float32)
    # convert color to BGR format for OpenCV
    rgb = cv2.cvtColor(segmentation_map, cv2.COLOR_RGB2BGR)
    mask_name= mask_name[0].split("/")[-1]
    cv2.imwrite(f"{folder}/e{epoch}_{mask_name}.png", segmentation_map)

    return segmentation_map
def draw_translucent_seg_maps(data, output, epoch, i, mask_name,folder):
    """
    This function color codes the segmentation maps that is generated while
    validating. THIS IS NOT TO BE CALLED FOR SINGLE IMAGE TESTING
    """
    alpha = 1 # how much transparency
    beta = 0.6 # alpha + beta should be 1
    gamma = 0 # contrast
    label_color_map = np.array([[0.0, 0.0, 0.0], [255.0, 0.0, 0.0]], dtype=np.float32)
    seg_map = output[0] # use only one output from the batch
    seg_map = torch.argmax(seg_map, dim=0).detach().cpu().numpy()

    image = data[0]
    image = np.array(image.cpu())
    image = np.transpose(image, (1, 2, 0))
    # unnormalize the image (important step)
    mean = np.array([0.0, 0.0, 0.0])
    std = np.array([1.0, 1.0, 1.0])
    image = std * image + mean
    image = np.array(image, dtype=np.float32)
    image = image * 255

    red_map = np.zeros_like(seg_map).astype(np.uint8)
    green_map = np.zeros_like(seg_map).astype(np.uint8)
    blue_map = np.zeros_like(seg_map).astype(np.uint8)


    for label_num in range(0, len(label_color_map)):
        index = seg_map == label_num
        red_map[index] = np.array(label_color_map)[label_num][0]
        green_map[index] = np.array(label_color_map)[label_num][1]
        blue_map[index] = np.array(label_color_map)[label_num][2]
        
    rgb = np.stack([red_map, green_map, blue_map], axis=2)
    rgb = np.array(rgb, dtype=np.float32)
    # convert color to BGR format for OpenCV
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    # cv2.imshow('rgb', rgb)
    # cv2.waitKey(0)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.addWeighted(image, alpha, rgb, beta, gamma, image)
    mask_name= mask_name[0].split("/")[-1]
    cv2.imwrite(f"{folder}/e{epoch}_b{i}_{mask_name}.png", image)

#Esta funcion probablemente no se utilice
def image_overlay(image, segmented_image):
    alpha = 1 # transparency for the original image 
    beta = 0.8 # transparency for the segmentation map
    gamma = 0 # scalar added to each sum
    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.addWeighted(image, alpha, segmented_image, beta, gamma, image)
    return image

def get_loaders(train_dir,train_maskdir,batch_size,transform_imag,transform_mask,num_workers,pin_memory=True):
    
    ds = IrisDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform_imag=transform_imag,
        transform_mask=transform_mask)
    
    n_val = int(len(ds) * 0.15)
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
        drop_last=True)

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
        drop_last=True
    )
    return train_loader, val_loader

def get_test(test_dir, test_maskdir,num_workers, test_transform_imag,test_transform_mask, pin_memory=True):
    test_ds = IrisDataset( 
        image_dir=test_dir,
        mask_dir=test_maskdir,
        transform_imag=test_transform_imag,
        transform_mask=test_transform_mask)
    test_loader= DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=pin_memory)
    names = os.listdir(test_dir)
    return test_loader, names

class save_best_model:
    def __init__(self, best_valid_loss=float("inf")):
        self.best_valid_loss = best_valid_loss
        self.best_valid_loss_epoch = float("inf")

    def __call__(self, current_valid_loss, epoch, model, optimizer, loss_fn):
        print(f"Current Best Validation Loss: ({self.best_valid_loss})", f"at epoch [{self.best_valid_loss_epoch}]")
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            self.best_valid_loss_epoch = epoch
            print(f"New Best Validation Loss: ({self.best_valid_loss})", f"at epoch [{self.best_valid_loss_epoch}]")
            torch.save({
                "epoch": epoch+1,
                "model_state_dict": model.state_dict(),
                "optimer_state_dict": optimizer.state_dict(),
                "loss": loss_fn,
                "best_model_epoch": self.best_valid_loss_epoch,
                "best_model_val": self.best_valid_loss,
                }, r"/home/jacobo15defrutos/AVS9/1.1-Unet3/Code/best_model_unet3.pth.tar")



def save_predictions_as_imgs(preds,tags,y,idx, folder="Data/saved_images/", device="cuda",testing=False, counter=0):
    if not testing:
        print("Saving validation images")
    else :
        print("Saving testing images")

    
    class_to_color=[torch.tensor([0.0,0.0,0.0], device='cuda'),torch.tensor([1.0,0.0,0.0], device='cuda')]
    output = torch.zeros(preds.shape[0], 3, preds.size(-2), preds.size(-1), dtype=torch.float) #Output size is set to preds.shape[0] as the size automatically changes to fit the remaining batch_size.
    for class_idx, color in enumerate(class_to_color):
        mask = preds[:,class_idx,:,:] == torch.max(preds, dim=1)[0]
        mask = mask.unsqueeze(1)
        curr_color = color.reshape(1, 3, 1, 1)
        segment = mask*curr_color 
        output += segment
    torchvision.utils.save_image(output, f"{folder}/{idx+1}_prediction.png")
#
    ##Save images to our saved_images folder
    #torchvision.utils.save_image(output, f"{folder}/{idx+1}_prediction.png")


            

def check_accuracy(loader, metrics,model, epoch, loss_fn,folder, device="cuda",show_individual_accuracy=False):
    print("-----Calculating Accuracy-----")
    model.eval()

    f1_score_iris_list=[]
    IoU_iris_list =[]   
    sens_iris_list = []
    PPV_iris_list=[]

    losses = []
    with torch.no_grad():
        for idx, sample in enumerate(iter(loader)):
            x = sample['image'].to(device=device)
            y = sample['mask'].long().to(device=device)
            y=y.squeeze(1)
            preds = model(x)
            outputs= preds#preds['out']
            #Calculate loss here
            loss = loss_fn(outputs, y)
            losses.append(loss.item())
            values,tags= torch.max(outputs,dim=1)
            y_pred = outputs.cpu().numpy().ravel()#añadir .data
            y_true = y.data.cpu().numpy().ravel()#añadir .data
            y_true2= y.cpu().numpy().ravel()#añadir .data
            for name, metric in metrics.items():
                #if name == 'f1_score':
                    # Use a classification threshold of 0.1
                    #f1_score_iris = metric(y_true==1,y_pred>0.02)
                if name== 'PPV':
                    PPV_iris = calc_PPV(tags, y.long())
                if name == 'IoU':
                    IoU_iris= calc_IoU(tags, y.long())
                elif name == 'Recall':
                    sens_iris = calc_sensitivity(tags, y.long())


            #Add accuracy to list to calculate the mean
            #Add the accuracy for each class to the mean calculator if the class is in the groundtruth.
            #If we don't do this then we add accuracies of 0% if the respective class isn't in the groundtruth image.
            if torch.sum(y == 1) > 0: 
                IoU_iris_list.append(IoU_iris)
                #f1_score_iris_list.append(f1_score_iris)
                sens_iris_list.append(sens_iris)
                PPV_iris_list.append(PPV_iris)
        
            #save_predictions_as_imgs(outputs,tags,y,idx)
            #draw_segmentation_map(outputs,sample['mask_name'],folder)
            draw_translucent_seg_maps(x,outputs,epoch,idx,sample['mask_name'],folder)
            #draw_segmentation_map(outputs)


        mean_IoU_iris = np.mean(IoU_iris_list)
        #mean_f1_score_iris = np.mean(f1_score_iris_list)
        mean_sens_iris= np.mean(sens_iris)
        mean_PPV_iris = np.mean(PPV_iris)
        
        
        print("     Iris IoU: ", mean_IoU_iris, "%")
        print("     Iris Sensitivity: ", mean_sens_iris, "%")
        print(f"    Iris PPV: {mean_PPV_iris} %")
        #print(f"    Iris F1_score: {mean_f1_score_iris} %")

        mean_loss = np.mean(losses)
        print("Validation Loss: ", mean_loss)
        
        #writer.add_scalar("Loss/val", mean_loss, epoch)
        #writer.close()

    model.train()
    return mean_loss
