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

        self.bin_loss = nn.DiceLoss(mode='binary')
        self.reg_loss = nn.MSELoss(reduction='mean')

    def forward(self,output,target):

        # Binary loss portion
        bin_out = output[:,0,:,:]
        bin_tar = target[:,0,:,:]
        bin_loss = self.bin_loss(bin_out.type(torch.long),bin_tar)

        # Regression loss portion
        reg_out = output[:,1,:,:]
        reg_tar = target[:,1,:,:]
        reg_loss = self.reg_loss(reg_out,reg_tar)

        return bin_loss, reg_loss


class MultiTaskModel(nn.Module):
    def __init__(self,multi_params):
        super().__init__()

        self.unet_model = multi_params['unet_model']        

        self.bin_active = nn.SoftMax()
        self.reg_active = nn.Sigmoid()
        
    def forward(self,x):

        # First pass through initial model w/ ImageNet pretraining
        x = self.unet_model(x)

        # Two different activations for output of model
        bin_out = self.bin_active(x)
        reg_out = self.reg_active(x)

        return torch.cat((bin_out,reg_out),dim=1)

"""
class combine_targets(Compose):
    def __call__(self,bin_target,reg_target):
        return np.concatenate((bin_target,reg_target),axis=-1)
"""
def combine_targets(bin_target, reg_target):
    catted = np.concatenate((np.expand_dims(bin_target,axis=2),np.expand_dims(reg_target,axis=2)),axis=-1)
    print(catted.shape)
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
            AlbuSeg2d(A.IAAPerspective(p=0.5)),
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
                                            )
        
        dataset_valid = SegmentationDataSet(inputs = valid_img_paths,
                                            targets = [valid_bin_tar,valid_reg_tar],
                                            transform = transforms_validation,
                                            use_cache = True,
                                            pre_transform = pre_transforms)

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
                                            pre_transform = pre_transforms)

    return dataset_train, dataset_valid


