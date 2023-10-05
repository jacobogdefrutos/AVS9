import torch
import torch.nn as nn
import torchvision.transforms as TF


# Convolution class containing the kernel size, stride and padding for our conv kernel.
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            #First convolution
            #Kernel size = 3 , stride = 1 , and padding = 1
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            #Second convolution
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

# The main UNET class. This class represents the UNET architecture with the input/output channels, features and the downsampling/upsampling part.
class UNET(nn.Module):
    def __init__( #CHANGE OUT_CHANNELS FROM 1 TO AMOUNT OF CLASSES WE NEED
        self, in_channels=3, out_channels=2, features=[64, 128, 256, 512],

    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)    

        #Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        #Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x) 
            skip_connections.append(x)
            x = self.pool(x)    #Apply max pooling operation

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            # this does not work atm
            if x.shape != skip_connection.shape:
                transform = TF.Resize(size=(skip_connection.shape[2:]))
                x = transform(x)

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)   # torch.sigmoid(x) if we want to use sigmoid instead 
