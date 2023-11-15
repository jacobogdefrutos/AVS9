import utils
from tqdm import tqdm
import torch.nn as nn
def train_one_epoch(model, trainloader, optimizer, epoch_idx,device):#tb_writer
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
        image = sample['image'][0].to(device)
        mask= sample['mask'][0].long().to(device)
        optimizer.zero_grad()
        pred, _ = model(image)
        #print(f'pred shape: {pred.shape}')
        mask = mask[0]
        total_mask = utils.get_totalmask(mask)
        pred = pred.to(device)
        #loss = utils.criterion(pred, total_mask,device)
        loss_fn= nn.CrossEntropyLoss()
        loss = loss_fn(pred, mask).requires_grad_()#tuve un error q decia expected scalar type Long, but found Byte
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    i = len(trainloader)
    last_loss = running_loss / i
    print(f'batch_loss for batch {i}: {last_loss}')
    tb_x = epoch_idx * len(trainloader) + i + 1
    #tb_writer.add_scalar('Loss/train', last_loss, tb_x)
    running_loss = 0.
    return last_loss