"""

Multi-Task Network model definition. Building off end layer of UNet++ model


"""

from Input_Pipeline import SegmentationDataSet
import torch
from torch import nn
import torch.nn.functional as F

import albumentations as A

from skimage.transform import resize
from Augmentation_Functions import *


class MultiTaskLoss(nn.Module):
    def __init__(self):
        super(MultiTaskLoss,self).__init__()

        self.reg_loss = nn.MSELoss(reduction='mean')
        #self.balance_term = nn.Parameter(torch.ones(1,requires_grad=True))
        self.balance_term = torch.ones(1,requires_grad=True)

    def dice_loss(self,input, target):
        smooth = 1

        iflat = input.view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()

        return 1-((2. *intersection+smooth)/(iflat.sum()+tflat.sum()+smooth))
    
    def combined_loss(self,bin_input, reg_input,target):
        combined_input = reg_input*(1-bin_input)

        return self.reg_loss(combined_input, target)


    def forward(self,output,target):

        # Binary loss portion
        bin_out = output[0,:,:]
        bin_tar = torch.squeeze(target[:,0,:,:])
        #print(f'Binary target size: {bin_tar.shape}')

        bin_loss = self.dice_loss(torch.round(bin_out).type(torch.long),torch.round(bin_tar).type(torch.long))
        # Regression loss portion
        reg_out = output[1,:,:]
        reg_tar = torch.squeeze(target[:,1,:,:])
        #print(f'Reg target size: {reg_tar.shape}')
        reg_loss = self.reg_loss(reg_out,reg_tar)

        #balanced_loss = ((1-self.balance_term)*bin_loss) + (self.balance_term*reg_loss)
        #print(f'Balance Term: {self.balance_term}')

        combined_loss = self.combined_loss(torch.round(bin_out).type(torch.int64),reg_out, reg_tar)

        return bin_loss, reg_loss, combined_loss


class MultiTaskModel(nn.Module):
    def __init__(self,multi_params):
        super().__init__()

        self.unet_model = multi_params['unet_model']        

        self.bin_active = nn.Sigmoid()
        self.reg_active = nn.Sigmoid()
    
    def forward(self,x):

        # First pass through initial model w/ ImageNet pretraining
        x = self.unet_model(x)
        #print(f'Output shape: {x.shape}')

        # Two different activations for output of model
        bin_out = self.bin_active(x[:,0,:,:])
        reg_out = self.reg_active(x[:,1,:,:])
        combined = torch.cat((bin_out,reg_out),dim=0)
        #print(f'Combined Shape: {combined.shape}')

        return combined
    
    def predict(self,x):
        x = self.forward(x)
        return x

"""
class combine_targets(Compose):
    def __call__(self,bin_target,reg_target):
        return np.concatenate((bin_target,reg_target),axis=-1)
"""
def combine_targets(bin_target, reg_target):
    catted = np.concatenate((np.expand_dims(bin_target,axis=2),np.expand_dims(reg_target,axis=2)),axis=-1)
    return catted


# Special make_training_set to deal with the multiple masks for each image
def make_multi_training_set(phase,train_img_paths,train_bin_tar,train_reg_tar,valid_img_paths,valid_bin_tar,valid_reg_tar):

    if phase=='train':

        pre_transforms = [
            ComposeTriple([
                FunctionWrapperTriple(combine_targets,
                                    input = False,
                                    target1 = True,
                                    target2 = True)
                    ]),
            ComposeDouble([
                FunctionWrapperDouble(resize,
                                    input = True,
                                    target = False,
                                    output_shape = (512,512,3),
                                    ),
                FunctionWrapperDouble(resize,
                                    input = False,
                                    target = True,
                                    output_shape = (512,512,2)
                                    )
                ])
        ]

        transforms_training = ComposeDouble([
            AlbuSeg2d(A.HorizontalFlip(p=0.5)),
            #AlbuSeg2d(A.IAAPerspective(p=0.5)),
            AlbuSeg2d(A.VerticalFlip(p=0.5)),
            FunctionWrapperDouble(np.moveaxis, input = True, target = True, source = -1, destination = 0),
            FunctionWrapperDouble(normalize_01,input = True, target = True)
        ])

        transforms_validation = ComposeDouble([
            FunctionWrapperDouble(np.moveaxis,input=True,target=True,source=-1,destination=0),
            FunctionWrapperDouble(normalize_01, input = True, target = True)
        ])

        dataset_train = SegmentationDataSet(inputs = train_img_paths,
                                            targets = [train_bin_tar,train_reg_tar],
                                            transform = transforms_training,
                                            use_cache = True,
                                            pre_transform = pre_transforms,
                                            target_type = ['binary','nonbinary'])
        
        dataset_valid = SegmentationDataSet(inputs = valid_img_paths,
                                            targets = [valid_bin_tar,valid_reg_tar],
                                            transform = transforms_validation,
                                            use_cache = True,
                                            pre_transform = pre_transforms,
                                            target_type = ['binary','nonbinary'])

    elif phase=='test':
        pre_transforms = [
            ComposeTriple([
                FunctionWrapperTriple(combine_targets,
                                    input = False,
                                    target1 = True,
                                    target2 = True)
                ]),
            ComposeDouble([
                FunctionWrapperDouble(resize,
                                    input = True,
                                    target = False,
                                    output_shape = (512,512,3),
                                    ),
                FunctionWrapperDouble(resize,
                                    input = False,
                                    target = True,
                                    output_shape = (512,512,2)
                                    )
                ])
        ]

        transforms_testing = ComposeDouble([
            FunctionWrapperDouble(np.moveaxis, iniput = True, target = True, source = -1, destination = 0),
            FunctionWrapperDouble(normalize_01,input = True, target = True)
        ])

        dataset_train = None

        dataset_valid = SegmentationDataSet(inputs = valid_img_paths,
                                            targets = [valid_bin_tar,valid_reg_tar],
                                            transform = transforms_testing,
                                            use_cache = True,
                                            pre_transform = pre_transforms,
                                            target_type = ['binary','nonbinary'])

    return dataset_train, dataset_valid


