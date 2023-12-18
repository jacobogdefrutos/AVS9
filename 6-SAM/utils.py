
from tqdm import tqdm
import re
import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score,jaccard_score,recall_score,precision_score
import torch
from dataset import IrisDataset
from torch.utils.data import DataLoader, random_split
import os

def get_test(test_dir, test_maskdir,num_workers, transform, pin_memory=True):
    test_ds = IrisDataset( 
        image_dir=test_dir,
        mask_dir=test_maskdir,
        transform = transform)
    test_loader= DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=pin_memory)
    names = os.listdir(test_dir)
    return test_loader, names

def get_loaders(train_dir,train_maskdir,batch_size,transform,num_workers,pin_memory=True):
    
    ds = IrisDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=transform,
        )
    
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
def get_totalmask(masks):
    """get all masks in to one image
    ARGS:
        masks (torch.Tensor): shape: (N, H, W) where N is the number of masks
                              masks H,W is usually 1024,1024
    RETURNS:
        total_gt (torch.Tensor): all masks in one image

    """
    total_gt = torch.zeros_like(masks[0,:,:])
    for k in range(len(masks)):
        total_gt += masks[k,:,:]
    return total_gt



class FocalLoss(nn.Module):
    """ Computes the Focal loss. """

    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):
        alpha=0.5
        gamma=1
        inputs = inputs.flatten(0,2)
        targets= targets.float()
        BCE = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')#cambiar
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1 - BCE_EXP)**gamma * BCE

        return focal_loss


class DiceLoss(nn.Module):
    """ Computes the Dice loss. """

    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = F.sigmoid(inputs)
        inputs = inputs.flatten(0,2)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / \
            (inputs.sum() + targets.sum() + smooth)
        return 1 - dice



def criterion(x, y,DEVICE):
    """ Combined dice and focal loss.
    ARGS:
        x: (torch.Tensor) the model output
        y: (torch.Tensor) the target
    RETURNS:
        (torch.Tensor) the combined loss

    """
    focal, dice = FocalLoss(), DiceLoss()
    y = y.to(DEVICE)
    x = x.to(DEVICE)
    return 20 * focal(x, y) + dice(x, y)
def val_loss(loader,model,transform,boxes_dic, epoch,folder, device="cuda"):#boxes_dic
    print("-----Validation data-----")
    IoU_iris_list =[]
    IoU_iris_list_sam=[]
    PPV_iris_list =[]
    Recall_iris_list =[]
    f1_score_iris_list=[]
    model.eval()
    running_vloss=0
    with torch.no_grad():
        for i, sample in enumerate(iter(loader)):
            ppv=0
            recall=0
            image = sample['image'].squeeze(1).to(device)
            mask= sample['mask'].squeeze(1).long().to(device)
            mask=mask[0]
            idx= sample['idx'].item()
            og_y,og_x= sample['original_image_size']
            original_image_size=(og_y.item(),og_x.item())
            prompt_box=boxes_dic[idx]['cords']
            #tensor_box= sample['box']
            #prompt_box = np.array([tensor.item() for tensor in tensor_box])
            box = transform.apply_boxes(prompt_box, original_image_size)
            box_torch = torch.as_tensor(box, dtype=torch.float, device=device)
            box_torch = box_torch[None, :]
            total_mask = get_totalmask(mask)
            total_mask = total_mask.to(device)
            preds, iou = model(image,box_torch)
            vloss = criterion(preds, total_mask,device)
            running_vloss += vloss.item()
            preds_prob = torch.sigmoid(preds.squeeze(1))# shape (1,1024,1024)
            preds_prob_numpy = preds_prob.cpu().numpy().squeeze()
            preds_binary = (preds_prob_numpy > 0.9).astype(np.uint8)
            if len(np.unique(preds_binary))>1:
                ppv=precision_score(total_mask.numpy(),preds_binary,average='micro')
                recall= recall_score(total_mask.numpy(),preds_binary,average='micro')
                iou_sklearn= jaccard_score(total_mask.numpy(),preds_binary,average='micro')
                f_score=f1_score(total_mask.numpy(),preds_binary,average='micro')
            else:
                ppv=0
                recall=0
                iou_sklearn=0
                f_score=0

            draw_translucent_seg_maps(image, preds_binary, epoch,idx,folder)
            IoU_iris_list_sam.append(iou.item())
            IoU_iris_list.append(iou_sklearn)
            PPV_iris_list.append(ppv)
            Recall_iris_list.append(recall)
            f1_score_iris_list.append(f_score)

        mean_IoU_iris_sam = np.mean(IoU_iris_list_sam)
        mean_IoU_iris = np.mean(IoU_iris_list)
        mean_PPV_iris = np.mean(PPV_iris_list)
        mean_Recall_iris = np.mean(Recall_iris_list)
        mean_f1_score_iris=np.mean(f1_score_iris_list)
        avg_vloss = running_vloss / len(loader)
        print(f'Epoch: {epoch}, validloss: {avg_vloss}')
        print("     Iris IoU from SAM: ", mean_IoU_iris_sam*100, "%")
        print("     Iris IoU sklearn: ", mean_IoU_iris*100, "%")
        print("     Irris PPV: ", mean_PPV_iris*100, "%")
        print("     Iris Recall: ", mean_Recall_iris*100, "%")
        print(f"    Iris F1_score: {mean_f1_score_iris*100} %")
    return avg_vloss
def find_number_in_string(input_string):
    # Regular expression to find a number in a string
    pattern = r'\d+'

    # Search for the pattern in the input string
    match = re.search(pattern, input_string)

    # Check if a match is found
    if match:
        # Extract and return the matched number
        return int(match.group())

    # Return None if no number is found
    return None
def draw_translucent_seg_maps(data, output,epoch, i,folder):
    """
    This function color codes the segmentation maps that is generated while
    validating. THIS IS NOT TO BE CALLED FOR SINGLE IMAGE TESTING
    """
    alpha = 1 # how much transparency
    beta = 0.6 # alpha + beta should be 1
    gamma = 0 # contrast
    label_color_map = np.array([[0.0, 0.0, 0.0], [255.0, 0.0, 0.0]], dtype=np.float32)
    seg_map = output # use only one output from the batch y queremos shape (H,W)
    #seg_map = torch.argmax(seg_map, dim=0).detach().cpu().numpy()
    image = data[0]# queremos formato (C,H,W)
    ## unnormalize the image (important step)
    #mean = np.array([0.0, 0.0, 0.0])
    #std = np.array([1.0, 1.0, 1.0])
    #image = std * image + mean
    image = np.array(image, dtype=np.float32)
    image = np.transpose(image, (1, 2, 0))
    #image = image * 255
    red_map = np.zeros_like(seg_map).astype(np.uint8)
    green_map = np.zeros_like(seg_map).astype(np.uint8)
    blue_map = np.zeros_like(seg_map).astype(np.uint8)


    for label_num in range(0, len(label_color_map)):
        index = seg_map == label_num
        red_map[index] = np.array(label_color_map)[label_num][0] #(H,W)
        green_map[index] = np.array(label_color_map)[label_num][1]
        blue_map[index] = np.array(label_color_map)[label_num][2]

    rgb = np.stack([red_map, green_map, blue_map], axis=2)#(H,W,C)
    rgb = np.array(rgb, dtype=np.float32)
    # convert color to BGR format for OpenCV
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    # cv2.imshow('rgb', rgb)
    # cv2.waitKey(0)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.addWeighted(image, alpha, rgb, beta, gamma, image)
    cv2.imwrite(f"{folder}/val_{epoch}_{i}.png", image)

class save_best_model:
    def __init__(self, best_valid_loss=float("inf")):
        self.best_valid_loss = best_valid_loss
        self.best_valid_loss_epoch = float("inf")

    def __call__(self, current_valid_loss, epoch, model, optimizer):
        print(f"Current Best Validation Loss: ({self.best_valid_loss})", f"at epoch [{self.best_valid_loss_epoch}]")
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            self.best_valid_loss_epoch = epoch
            print(f"New Best Validation Loss: ({self.best_valid_loss})", f"at epoch [{self.best_valid_loss_epoch}]")
            torch.save({
                "epoch": epoch+1,
                "model_state_dict": model.state_dict(),
                "optimer_state_dict": optimizer.state_dict(),
                #"loss": loss_fn,
                "best_model_epoch": self.best_valid_loss_epoch,
                "best_model_val": self.best_valid_loss,
                }, r"/home/jgonzafrutos/AVS9/6-SAM/best_model_yoloSAM_25_epochs_0001_tol09_g1.pth.tar")
        
        