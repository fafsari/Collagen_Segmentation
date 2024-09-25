
"""
Testing the Data input pipeline 

from: https://towardsdatascience.com/creating-and-training-a-u-net-model-with-pytorch-for-2d-3d-semantic-segmentation-dataset-fb1f7f80fe55
"""

import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.io import imread
from math import floor
import cv2

import torch
from torch.utils.data import Dataset
import albumentations

from skimage.transform import resize

from Augmentation_Functions import *
from CollagenSegUtils import resize_special
from CollagenSegUtils import visualize_continuous

# Input class with required len and getitem functions
# in this case, the inputs and targets are lists of paths matching paths for image and segmentation ground-truth
# *** Have to make sure the input and target paths are aligned ***
# Need to change shape of input data to be (batch size, channels, height, width)
# with linear scaling depending on network expectations,
# target to be (batch size, height, width) with dense integer encoding for each class

class SegmentationDataSet(Dataset):
    def __init__(self,
                 inputs: list,
                 targets: list,
                 transform = None,
                 pre_transform = None,
                 target_type = None,
                 batch_size = None,
                 parameters = {},
                 phase = None):
        

        self.inputs = inputs
        self.targets = targets
        self.transform = transform
        self.inputs_dtype = torch.float32
        self.parameters = parameters
        self.phase = phase

        if target_type == 'binary':
            self.targets_dtype = torch.long
        elif target_type == 'nonbinary':
            self.targets_dtype = torch.float32
        elif target_type is None:
            self.targets_dtype = torch.float32
        
        self.batch_size = batch_size
        self.patch_batch = False
        self.sample_weight = False

        self.testing_metrics = False if len(self.targets)==0 else True

        # increasing dataset loading efficiency
        self.pre_transform = pre_transform
        
        self.cached_data = []
        self.cached_names = []
        self.image_means = []
        self.image_stds = []
        self.images = []
        self.cached_item_names = []
        self.item_idx = -1
        self.cached_item_index = 0

        image_size = [int(i) for i in self.parameters['preprocessing']['image_size'].split(',')]
        
        progressbar = tqdm(range(len(self.inputs)), desc = 'Caching')

        if len(self.targets)>0:
            # For a training round or testing round with ground truth labels available
            for i, img_name, tar_name in zip(progressbar, self.inputs, self.targets):
                try:
                    # For multi-channel image inputs
                    if type(img_name)==list:
                        img1,img2,tar = imread(str(img_name[0])), imread(str(img_name[1])),imread(str(tar_name))
                        img = np.concatenate((img1,img2),axis=-1)
                        img_name = img_name[0]
                    else:
                        # img, tar = imread(str(img_name)), imread(str(tar_name))
                        img, tar = cv2.cvtColor(cv2.imread(str(img_name), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB), cv2.imread(str(tar_name), cv2.IMREAD_GRAYSCALE)

                    self.images.append((img,tar))
                    self.cached_item_names.append(img_name)
                except FileNotFoundError:
                    print(f'File not found: {img_name}, {tar_name}')
        
        else:
            # For just predicting on images with no ground truth provided
            for i, img_name in zip(progressbar,self.inputs):
                try:
                    if type(img_name)==list:
                        img1,img2 = imread(str(img_name[0])),imread(str(img_name[1]))
                        img = np.concatenate((img1,img2),axis=-1)
                        img_name = img_name[0]
                    else:
                        # img = imread(str(img_name))
                        img = cv2.cvtColor(cv2.imread(str(img_name), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                    
                    tar = np.zeros((np.shape(img)[0],np.shape(img)[1],1))

                    self.images.append((img,tar))
                    self.cached_item_names.append(img_name)

                except FileNotFoundError:
                    print(f'File not found: {img_name}')
        
        # For images that are smaller/same size as the model's patch size then just resize as normal   
        for (img, tar),name in tqdm(zip(self.images,self.cached_item_names),total=len(self.images),desc = 'Preprocessing Images'):     
            
            if self.pre_transform is not None:
                    img, tar = self.pre_transform(img, tar)
                    
            if np.shape(img)[0]<=image_size[0] and np.shape(img)[1]<=image_size[1]:
           
                if self.pre_transform is not None:
                    img, tar = self.pre_transform(img, tar)

                # Calculating dataset mean and standard deviation
                img_channel_mean = np.mean(img,axis=(0,1))
                img_channel_std = np.std(img,axis=(0,1))

                self.image_means.append(img_channel_mean)
                self.image_stds.append(img_channel_std)
                
                # Data used for training and testing
                self.cached_data.append((img,tar))
                self.cached_names.append(name)
            
            else:

                # Overlap percentage, hardcoded patch size
                self.patch_size = [image_size[0],image_size[1]]
                self.patch_batch = 0.05
                # Correction for downsampled (10X as opposed to 20X) data
                self.downsample_level = 0.5
                stride = [int(self.patch_size[0]*(1-self.patch_batch)*self.downsample_level), int(self.patch_size[1]*(1-self.patch_batch)*self.downsample_level)]

                # Calculating and storing patch coordinates for each image and reading those regions at training time :/
                n_patches = [1+floor((np.shape(img)[0]-self.patch_size[0])/stride[0]), 1+floor((np.shape(img)[1]-self.patch_size[1])/stride[1])]
                start_coords = [0,0]

                row_starts = [int(start_coords[0]+(i*stride[0])) for i in range(0,n_patches[0])]
                col_starts = [int(start_coords[1]+(i*stride[1])) for i in range(0,n_patches[1])]
                row_starts.append(int(np.shape(img)[0]-self.patch_size[0]))
                col_starts.append(int(np.shape(img)[1]-self.patch_size[1]))
                
                self.original_image_size = list(np.shape(img))

                # Iterating through the row_starts and col_starts lists and applying pre_transforms to the image
                item_patches = []
                patch_names = []
                for r_s in row_starts:
                    for c_s in col_starts:
                        new_img = img[r_s:r_s+self.patch_size[0], c_s:c_s+self.patch_size[1],:]
                        new_tar = np.zeros((np.shape(new_img)[0],np.shape(new_img)[1]))

                        if self.pre_transform is not None:
                            new_img, new_tar = self.pre_transform(new_img, new_tar)

                        ############################################ Added for Ingrowth project
                        if self.phase == 'train':
                            self.cached_data.append((new_img,new_tar))
                            self.cached_names.append(name.replace(f'.{name.split(".")[-1]}',f'_{r_s}_{c_s}.{name.split(".")[-1]}'))   
                        ############################################ Commented for Ingrowth project
                        else:
                            item_patches.append((new_img,new_tar))
                            patch_names.append(name.replace(f'.{name.split(".")[-1]}',f'_{r_s}_{c_s}.{name.split(".")[-1]}'))

                
                if self.phase == 'test':
                    self.cached_data.append(item_patches)
                    self.cached_names.append(patch_names)
                
                    self.cached_item_patches = [len(i) for i in self.cached_data]                
                ##########################################################################

        print(f'Cached Data: {len(self.cached_data)}')

    def __len__(self):
        ############################################ Added for Ingrowth project
        if self.phase == 'train':
            return len(self.cached_data)
        ############################################ Commented for Ingrowth project
        else:
            if not self.patch_batch:
                return len(self.cached_data)
            else:
                return sum([len(i) for i in self.cached_data])
        
    # Getting matching input and target(label)
    def __getitem__(self,index:int):
        #index = self.item_idx
        ############################################ Added for Ingrowth project
        if self.phase == 'train':
            x, y = self.cached_data[index]
            input_ID = self.cached_names[index]
            
            # Preprocessing steps (if there are any)
            if self.transform is not None:
                x, y = self.transform(x, y)
            
            # Adding batch dimension
            # x = x[None,:,:,:]
            # y = y[None,:,:,:]
            
            # Getting in the right input/target data types
            x, y = torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y).type(self.targets_dtype)
            
            return x, y, input_ID
        
        ############################################ Commented for Ingrowth project
        if not self.patch_batch:
            x, y = self.cached_data[index]
            input_ID = self.cached_names[index]
            
            # Preprocessing steps (if there are any)
            if self.transform is not None:
                x, y = self.transform(x, y)
            
            # Adding batch dimension
            #x = x[None,:,:,:]
            #y = y[None,:,:,:]
            
            # Getting in the right input/target data types
            x, y = torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y).type(self.targets_dtype)

            return x, y, input_ID

        else:

            # Just returning a list of all patches for this index:
            image_patches = self.cached_data[index]
            patch_names = self.cached_names[index]

            x_list = []
            y_list = []
            input_ID_list = []
            for img_tar_pair, patch_id in zip(image_patches,patch_names):
                img, tar = img_tar_pair

                if self.transform is not None:
                    x, y = self.transform(img, tar)
                
                # Adding batch dimension
                x = x[None,:,:,:]
                y = y[None,:,:,:]
                x, y = torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y).type(self.targets_dtype)
                x_list.append(x)
                y_list.append(y)
                input_ID_list.append(patch_id)

            return x_list, y_list, input_ID_list
        
def make_training_set(phase,train_img_paths, train_tar, valid_img_paths, valid_tar,parameters):
    
    img_size = [int(i) for i in parameters['preprocessing']['image_size'].split(',')]
    mask_size = [int(i) for i in parameters['preprocessing']['mask_size'].split(',')]
    color_transform = parameters['preprocessing']['color_transform']

    if 'image_means' in parameters['preprocessing']:
        image_means = [float(i) for i in parameters['preprocessing']['image_means'].split(',')]
        image_stds = [float(i) for i in parameters['preprocessing']['image_stds'].split(',')]
    else:
        # Normalization is normally subtracting mean and dividing by standard deviation
        # Setting the image_means to 0 and stds to 1 nullifies the effect
        image_means = [float(0.0) for i in range(img_size[-1])]
        image_stds = [float(1.0) for i in range(img_size[-1])]
        
        pre_transforms = ComposeDouble([
                FunctionWrapperDouble(resize_special,
                                    input = True,
                                    target = False,
                                    output_size = img_size,
                                    transform = color_transform),
                FunctionWrapperDouble(resize,
                                    input = False,
                                    target = True,
                                    output_shape = mask_size),
                FunctionWrapperDouble(normalize,
                                      input = True,
                                      target = False,
                                      mean = image_means,
                                      std = image_stds)
        ])        

        # Continuous target type augmentations
        transforms_training = ComposeDouble([
            AlbuSeg2d(albumentations.HorizontalFlip(p=0.5)),
            AlbuSeg2d(albumentations.RandomBrightnessContrast(p=0.2)),
            AlbuSeg2d(albumentations.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.5,rotate_limit=45,interpolation=1,p=0.1)),
            AlbuSeg2d(albumentations.VerticalFlip(p=0.5)),
            FunctionWrapperDouble(np.moveaxis, input=True, target=True, source=-1, destination=0)
        ])

        transforms_validation = ComposeDouble([
            FunctionWrapperDouble(np.moveaxis,input=True, target=True, source=-1, destination=0)
        ])
        if parameters["mode"] == "None":
            dataset_train = SegmentationDataSet(inputs = train_img_paths,
                                             targets = train_tar,
                                             transform = transforms_training,
                                             pre_transform = pre_transforms,
                                             parameters = parameters,
                                             phase=phase)
        elif parameters["mode"] == "ADDA":
            dataset_train = SegmentationDataSet(inputs = train_img_paths,
                                                targets = train_tar,
                                                transform = transforms_validation,
                                                pre_transform = pre_transforms,
                                                parameters = parameters,
                                                phase=phase)
        
        dataset_valid = SegmentationDataSet(inputs = valid_img_paths,
                                             targets = valid_tar,
                                             transform = transforms_validation,
                                             pre_transform = pre_transforms,
                                             parameters = parameters,
                                             phase=phase)
    return dataset_train, dataset_valid


xls = pd.ExcelFile("/orange/pinaki.sarder/f.afsari/5-Ingrowth_Brain/Data/source_train_test.xlsx")
sheets = {sheet: pd.read_excel(xls, sheet_name=sheet) for sheet in xls.sheet_names}
for sheet, df in sheets.items():
    print(sheet, df.columns)

train_images = [image_name for image_name in sheets['Train']['train_images']]
train_labels = [image_name for image_name in sheets['Train']['train_labels']]

test_images = [image_name for image_name in sheets['Test']['test_images']]
test_labels = [image_name for image_name in sheets['Test']['test_labels']]


training_parameters = {	
		"mode": "None",
        "model_details":{				
            "architecture":"Unet++",
            "encoder":"resnet34",
            "encoder_weights":"imagenet",
			"dropout_rate":0.4,
            "active":"sigmoid",
            "target_type":"nonbinary",
            "in_channels":3,
            "ann_classes":"background,ingrowth"
        },
        "train_test_split":{
            "split":0.8,
            "train_test_source": "/orange/pinaki.sarder/f.afsari/5-Ingrowth_Brain/Data"
        },
        "loss":"MSE",
        "lr":0.00005,
        "batch_size":2,
        "step_num":4000,
        "save_step":20,
        "preprocessing":{
            "image_size":"512,512,3",
            "mask_size":"512,512,1",
            "color_transform":"None"
        }
}

dataset_train, dataset_valid = make_training_set(
    'train',
    train_img_paths = train_images,
    train_tar = train_labels,
    valid_img_paths=test_images,
    valid_tar=test_labels,
    parameters= training_parameters
)

batch_size = 2
train_loader = DataLoader(dataset_train, batch_size = batch_size, shuffle = True, num_workers = 4)
valid_loader = DataLoader(dataset_valid, batch_size = batch_size, shuffle = True, num_workers = 4)

train_iter = iter(train_loader)
val_iter = iter(valid_loader)      

train_batch = next(train_iter)
valid_batch = next(val_iter)

# first_batch = torch.load("train_val_loader_test.pt")
first_batch = {
        'train_batch': train_batch,
        'valid_batch': valid_batch
    }
print()
for k, v in first_batch.items():
    
    img_dict = {'Image':v[0][0], 'Pred_Mask':v[1][0],'Ground_Truth':v[1][0]}

    fig = visualize_continuous(img_dict,'comparison')
    image_name = v[2][0].split('/')[-1]
    fig.savefig(f'/orange/pinaki.sarder/f.afsari/5-Ingrowth_Brain/Data/Tests/{image_name}')
               
    print(k, v[0][0].shape, v[1][0].shape, v[2][0])
    
print()