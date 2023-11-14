import torch
import math
# Calculates the Intersection over Union of prediction (tags) compared to the ground truth (mask)
def calc_IoU(tags, mask):
    #IoU (jaccard_similarity_score)=   True Positives               / (True Positives              +  False Positive              + False Negative )
    mask= mask.squeeze(1)
    IoU_iris = (torch.sum(tags[mask == 1] == 1) / (torch.sum(tags[mask == 1] == 1) +  torch.sum(tags[mask != 1] == 1) + torch.sum(tags[mask == 1] != 1))*100).item()
    #In specific cases where ground truth does not have a class that is predicted we get a NaN error as we try to divide by 0
    if math.isnan(IoU_iris): IoU_iris = 0
    return IoU_iris

# Sensitivity also known as Recall
def calc_sensitivity(tags, mask): 
    mask=mask.squeeze(1)   
    #Sens =          True Positives               / (True Positives                                + False Negative )
    sens_iris= (torch.sum(tags[mask == 1] == 1) / (torch.sum(tags[mask == 1] == 1)  + torch.sum(tags[mask == 1] != 1))*100).item()
    #In specific cases where ground truth does not have a class that is predicted we get a NaN error as we try to divide by 0
    if math.isnan(sens_iris): sens_iris = 0
    return sens_iris

def calc_PPV(tags, mask):
    #PPV =          True Positives               / (True Positives                                + False Positive )
    mask=mask.squeeze(1)
    PPV_iris= (torch.sum(tags[mask == 1] == 1) / (torch.sum(tags[mask == 1] == 1) + torch.sum(tags[mask != 1] == 1))*100).item()
    #In specific cases where ground truth does not have a class that is predicted we get a NaN error as we try to divide by 0
    if math.isnan(PPV_iris): PPV_iris = 0
    
    return PPV_iris