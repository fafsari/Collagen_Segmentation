"""
Fusion model leveraging different modes of images available

UNet base architecture from: https://github.com/milesial/Pytorch-UNet/blob/master/unet

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Parts of UNet

"""

class DoubleConv(nn.Module):
    def __init__(self,in_channels,out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        dropout = None

        if dropout is not None:
            self.dropout = dropout

            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3,padding=1,bias=False),
                nn.Dropout(p=self.dropout),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels,out_channels,kernel_size=3,padding=1,bias=False),
                nn.Dropout(p=self.dropout),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

            self.double_conv_3d = nn.Sequential(
                nn.Conv3d(in_channels,mid_channels,kernel_size=3,padding=1,bias=False),
                nn.Dropout(p=self.dropout),
                nn.BatchNorm3d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(mid_channels,1,kernel_size=3,padding=1,bias=False),
                nn.Dropout(p=self.dropout),
                nn.BatchNorm3d(1),
                nn.ReLU(inplace=True)
            )
        else:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3,padding=1,bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels,out_channels,kernel_size=3,padding=1,bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

            self.double_conv_3d = nn.Sequential(
                nn.Conv3d(in_channels,mid_channels,kernel_size=3,padding=1,bias=False),
                nn.BatchNorm3d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(mid_channels,1,kernel_size=3,padding=1,bias=False),
                nn.BatchNorm3d(1),
                nn.ReLU(inplace=True)
            )

    def forward(self,x):
        
        x = self.double_conv(x)

        return x


class Down(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels,out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
    

class Up(nn.Module):
    def __init__(self,in_channels,out_channels,bilinear=True):
        super().__init__()

        # Use normal convolutions if bilinear
        if bilinear:
            self.up = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
            self.conv = DoubleConv(in_channels,out_channels,in_channels//2)

        else:
            in_channels = int(in_channels)
            out_channels = int(out_channels)
            self.up = nn.ConvTranspose2d(in_channels,in_channels//2,kernel_size=2,stride=2)
            self.conv = DoubleConv(in_channels,out_channels)

    def forward(self,x1,x2):

        x1 = self.up(x1)
        # Input in the form channelsXheightXwidth
        diffY = x2.size()[2]-x1.size()[2]
        diffX = x2.size()[3]-x1.size()[3]

        x1 = F.pad(x1, [diffX//2, diffX-diffX//2, diffY//2, diffY-diffY//2])
        # references for padding issues:
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2,x1],dim=1)
        return self.conv(x)
    
class OutConv(nn.Module):
    def __init__(self,in_channels,out_channels,final_active = None):
        super(OutConv,self).__init__()
        self.conv = nn.Conv2d(int(in_channels),int(out_channels),kernel_size=1)

        if final_active:
            if final_active=='sigmoid':
                self.final_active = nn.Sigmoid()
        else:
            self.final_active = nn.Identity()

    def forward(self, x):

        x = self.conv(x)
        x = self.final_active(x)

        return x
    

"""
Synthesizing into full UNet architecture

"""

class UNet(nn.Module):
    def __init__(self,n_channels,n_classes,bilinear=False):
        super(UNet,self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64,128))
        self.down2 = (Down(128,256))
        self.down3 = (Down(256,512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512,1024//factor))
        self.up1 = (Up(1024,512//factor,bilinear))
        self.up2 = (Up(512,256//factor,bilinear))
        self.up3 = (Up(256,128//factor,bilinear))
        self.up4 = (Up(128,64,bilinear))
        self.outc = (OutConv(64,n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5,x4)
        x = self.up2(x,x3)
        x = self.up3(x,x2)
        x = self.up4(x,x1)
        logits = self.outc(x)
        return logits
    

class DUNet(nn.Module):
    def __init__(self,
                 n_channels: list,
                 n_classes: int,
                 activation: str,
                 bilinear=False):
        super().__init__()

        # In this case we can make n_channels a list and modify it for each image type (ex: we could do 3 channels of brightfield and 1 channel of DUET, etc.)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.activation = activation
        self.bilinear = bilinear
        self.down_n = 4
        self.up_n = self.down_n
        self.in_out_channel = 64
        self.factor = 2 if self.bilinear else 1
        
        # Down layers
        self.input1 = (DoubleConv(self.n_channels[0],self.in_out_channel,mid_channels=3))
        self.input2 = (DoubleConv(self.n_channels[1],self.in_out_channel,mid_channels=3))
        for i in range(self.down_n):

            #print(f'd{i}: in: {(2**i)*self.in_out_channel}, out: {(2**(i+1))*self.in_out_channel}')
            if not i==self.down_n:
                setattr(self,f'down{i}',(Down((2**i)*self.in_out_channel,(2**(i+1))*self.in_out_channel)))
            else:
                setattr(self,f'down{i}',(Down((2**i)*self.in_out_channel,(2**(i+1))*self.in_out_channel//self.factor)))

        # Up layers
        start_dim = (2**(i+1)*self.in_out_channel//self.factor)*2
        for i in range(self.up_n):
            #print(f'up{i}, in: {(1/(2**i))*start_dim}, out: {(1/(2**(i+1)))*start_dim//self.factor}')
            setattr(self,f'up{i}',(Up((1/(2**i))*start_dim,(1/(2**(i+1)))*start_dim//self.factor,self.bilinear)))
        
        # Prediction output
        last_dim = (1/(2**(i+1)))*start_dim//self.factor
        self.output = (OutConv(last_dim,self.n_classes,final_active=activation))

    def forward(self, x):

        # For multi-modal input: 'x' is a list of two different modes (batchXchannelsXheightXwidth)
        #print(f'input shape: {x.size()}')
        input_list = [x[:,0:self.n_channels[0],:,:],x[:,self.n_channels[0]:,:,:]]
        #print(f'input_list: {input_list[0].size()},{input_list[1].size()}')
        input_conv = [self.input1(input_list[0]),self.input2(input_list[1])]
        #print(f'input_conv: {input_conv[0].size()}, {input_conv[1].size()}')
        setattr(self,'d0',torch.cat(input_conv,dim=1))

        for d in range(1,self.down_n+1):
            #print(f'd: {d}')
            if d==1:
                down1 = getattr(self,f'down{d-1}')(input_conv[0])
                down2 = getattr(self,f'down{d-1}')(input_conv[1])
            else:
                down1 = getattr(self,f'down{d-1}')(down1)
                down2 = getattr(self,f'down{d-1}')(down2)

            #print(f'donw1 shape: {down1.size()}')
            #print(f'down2 shape: {down2.size()}')

            setattr(self,f'd{d}',torch.cat([down1,down2],dim=1))

        for u in range(self.up_n):
            #print(f'u:{u}, using down{self.down_n-(u+1)}, shape: {getattr(self,f"d{self.down_n-(u+1)}").size()}')
            
            if u==0:
                up = getattr(self,f'up{u}')(getattr(self,f'd{self.down_n}'),getattr(self,f'd{self.down_n-(u+1)}'))
            else:
                up = getattr(self,f'up{u}')(up,getattr(self,f'd{self.down_n-(u+1)}'))
            
            #print(f'up shape: {up.size()}')

        pred = self.output(up)

        return pred

