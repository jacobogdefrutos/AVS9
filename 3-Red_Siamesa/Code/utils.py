from dataset import SiameseDataset
import torch
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
        shuffle=False,
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
def train_fn(loader, model, optimizer, loss_fn,scaler,device):
    model.to(device)
    losses = []
    model.train()
    for i,sample in tqdm(enumerate(loader)):
        img_OI,img_OD, label =sample
        optimizer.zero_grad()
        #forward
        with torch.set_grad_enabled(True):
            prediction1,prediction2 = model(img_OI,img_OD)
            loss = loss_fn(prediction1,prediction2, label)