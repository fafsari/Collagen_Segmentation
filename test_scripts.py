import torch
import segmentation_models_pytorch as smp

class EnsembleModelMIT(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 active,
                 n_classes):
        super().__init__()

        self.in_channels = in_channels
        self.active = active
        self.n_classes = n_classes

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        encoder = 'mit_b5'
        encoder_weights = 'imagenet'

        if self.active=='sigmoid':
            self.final_active = torch.nn.Sigmoid()
        elif self.active =='softmax':
            self.final_active = torch.nn.Softmax(dim=1)       
        elif self.active == 'linear':
            self.active = None
            self.final_active = torch.nn.Identity()
        else:
            self.active = None
            self.final_active = torch.nn.ReLU()

        self.model_b = smp.Unet(
                encoder_name = encoder,
                encoder_weights = encoder_weights,
                in_channels = int(self.in_channels/2),
                classes = self.n_classes,
                activation = self.active
                )
        
        self.model_d = smp.Unet(
            encoder_name = encoder,
            encoder_weights = encoder_weights,
            in_channels = int(self.in_channels/2),
            classes = self.n_classes,
            activation = self.active
        )

        self.combine_layers = torch.nn.Sequential(
            torch.nn.LazyConv2d(64,kernel_size=1),
            torch.nn.Dropout(p=0.1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64,self.n_classes,kernel_size=1)
        )

    def forward(self,input):

        b_input = input[:,0:int(self.in_channels/2),:,:]
        d_input = input[:,int(self.in_channels/2):self.in_channels,:,:]
        b_output = self.model_b.decoder(*self.model_b.encoder(b_input))
        d_output = self.model_d.decoder(*self.model_d.encoder(d_input))

        self.features = torch.cat((b_output,d_output),dim=1)
        final_prediction = self.final_active(self.combine_layers(self.features))
        
        return final_prediction

model_details = {
            "architecture":"ensembleMIT",
            "encoder":"mit_b5",
            "encoder_weights":"imagenet",
            "active":"sigmoid",
            "target_type":"nonbinary",
            "in_channels":6,
            "ann_classes":"background,collagen"
        }

source_model = EnsembleModelMIT(
            in_channels = model_details['in_channels'],
            active = model_details['active'],
            n_classes = 1
            )
target_model = EnsembleModelMIT(
            in_channels = model_details['in_channels'],
            active = model_details['active'],
            n_classes = 1
            )

model_path = "/blue/pinaki.sarder/f.afsari/4-DUET/Data/Results/Ensemble_MIT_RGB/models/Collagen_Seg_Model_Latest.pth"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:", device)

pretrained_dict = torch.load(model_path)

# Print the keys of the pre-trained state dictionary
print("Pre-trained model state dict keys:")
key_cnt = len(pretrained_dict.keys())
# for key in pretrained_dict.keys():
#     # print(key)
#     key_cnt += 1
print("Number of Pretrained keys:", key_cnt)

key_cnt_s = len(source_model.keys())
# for key in source_model.keys():
#     # print(key)
#     key_cnt_s += 1
print("Number of Source Model keys:", key_cnt_s)

source_model.load_state_dict(torch.load(model_path, map_location=device))

# Update target model's state_dict with source model's parameters
source_state_dict = source_model.state_dict()
target_state_dict = target_model.state_dict()
target_state_dict.update(source_state_dict)

# Load the updated state_dict into target model
target_model.load_state_dict(target_state_dict)

for param in target_model.parameters():
    param.detach_()

source_model.to(device)
target_model.to(device)


