from dataset import SiameseDataset
import torch
import numpy as np
from torch.utils.data import DataLoader
import tqdm


def get_loaders(train_csv,val_csv,train_dir,val_dir,batch_size,transform_imag,num_workers,pin_memory=True):
    train_ds = SiameseDataset(
        file_csv=train_csv,
        files_dir=train_dir,
        transform=transform_imag)
    val_ds = SiameseDataset(
        file_csv=val_csv,
        files_dir=val_dir,
        transform=transform_imag)

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
        shuffle=True,
        drop_last=True
    )
    return train_loader, val_loader
class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on:
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, x0, x1, y):
        # euclidian distance
        diff = x0 - x1
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq)

        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)
        loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
        loss = torch.sum(loss) / 2.0 / x0.size()[0]
        return loss
def train_fn(loader, model, optimizer, loss_fn,device):
    model.to(device)
    model.train()
    for i,sample in enumerate(loader,0):
        img_OI,img_OD, label =sample
        img_OI, img_OD, label = img_OI.to(device=device), img_OD.to(device=device), label.to(device=device)
        optimizer.zero_grad()
        #forward
        with torch.set_grad_enabled(True):
            prediction1,prediction2 = model(img_OI,img_OD)
            loss = loss_fn(prediction1,prediction2, label)
            loss.backward()
            optimizer.step()
    
    print("Training Loss: ", loss.item())
    return loss.item()
def val_loss(loader,model,optimizer,loss_fn,device):
    print("-----Calculating Validation loss-----")
    model.eval()
    with torch.no_grad():
        for i,sample in enumerate(loader,0):
            img_OI,img_OD, label =sample
            img_OI, img_OD, label = img_OI.to(device=device), img_OD.to(device=device), label.to(device=device)
            prediction1,prediction2 = model(img_OI,img_OD)
            loss = loss_fn(prediction1,prediction2, label)
    
        print("Validation Loss: ", loss.item())
    return loss.item()
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
                }, r"/home/jacobo15defrutos/AVS9_1/1-Unet/Code/saved_models/best_model_LeNet5RS.pth.tar")