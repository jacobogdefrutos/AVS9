import utils
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
def train_one_epoch(model, trainloader,transform,boxes_dic, optimizer, epoch_idx,device):#tb_writer,boxes_dic
    """ Runs forward and backward pass for one epoch and returns the average
    batch loss for the epoch.
    ARGS:
        model: (nn.Module) the model to train
        trainloader: (torch.utils.data.DataLoader) the dataloader for training
        optimizer: (torch.optim.Optimizer) the optimizer to use for training
        epoch_idx: (int) the index of the current epoch
        tb_writer: (torch.utils.tensorboard.writer.SummaryWriter) the tensorboard writer
    RETURNS:
        last_loss: (float) the average batch loss for the epoch

    """
    running_loss = 0.
    for  sample in tqdm(iter(trainloader)):
        image = sample['image'].squeeze(1).to(device)
        mask= sample['mask'].squeeze(1).long().to(device)
        idx= sample['idx'].item()
        og_y,og_x= sample['original_image_size']
        original_image_size=(og_y.item(),og_x.item())
        prompt_box=boxes_dic[idx]['cords']
        #tensor_box= sample['box']
        #prompt_box = np.array([tensor.item() for tensor in tensor_box])
        box = transform.apply_boxes(prompt_box, original_image_size)
        box_torch = torch.as_tensor(box, dtype=torch.float, device=device)
        box_torch = box_torch[None, :]
        optimizer.zero_grad()
        pred, _ = model(image,box_torch)
        #print(f'pred shape: {pred.shape}')
        mask = mask[0]
        total_mask = utils.get_totalmask(mask)
        pred = pred.to(device)
        loss = utils.criterion(pred, total_mask,device)
        #loss_fn= nn.CrossEntropyLoss()
        #loss = loss_fn(pred, mask).requires_grad_()#tuve un error q decia expected scalar type Long, but found Byte
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    i = len(trainloader)
    last_loss = running_loss / i
    print(f'Epoch: {epoch_idx}, Training loss: {last_loss}')
    #tb_x = epoch_idx * len(trainloader) + i + 1
    #tb_writer.add_scalar('Loss/train', last_loss, tb_x)
    running_loss = 0.
    return last_loss