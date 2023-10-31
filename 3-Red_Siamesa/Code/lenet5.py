import torch
import torch.nn as nn
import torchvision.transforms as TF

class LeNet5(nn.Module):
    #CHANGE OUT_CHANNELS FROM 1 TO AMOUNT OF CLASSES WE NEED
    def __init__( self, input_channels=1,num_classes=2):
        super(LeNet5, self).__init__()
        self.cnn1= nn.Sequential(
            nn.Conv2d(input_channels,6,kernel_size=5,bias= False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(6,16,kernel_size=5,bias= False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier= nn.Sequential(
            nn.Linear(16*53*53,120),
            nn.ReLU(inplace=True),
            nn.Linear(120,84),
            nn.ReLU(inplace=True),
            nn.Linear(84,num_classes),
            #nn.Softmax(dim=1)
            
        )
    def forward_once(self, x):
        # Forward pass
        x = self.cnn1(x)
        x = x.view(x.size()[0], -1)#it makes to 2, 16*132*199
        x = self.classifier(x)
        return x
    def forward(self,input1,input2):
        # forward pass of input 1
        output1 = self.forward_once(input1)
        # forward pass of input 2
        output2 = self.forward_once(input2)
        return output1, output2