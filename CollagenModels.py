import torch
import torch.nn.init as init
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.base.modules import Activation
from torchvision.ops import SqueezeExcitation


class SEBlock(torch.nn.Module):
    '''
    SEBlock Class: This class implements the Squeeze-and-Excitation block. 
        1- It reduces the channel dimensions, 
        2- Applies non-linearity, and 
        3- Restores the channel dimensions with a sigmoid activation to get channel-wise attention.
    '''
    def __init__(self, channels, reduction=8):
        super(SEBlock, self).__init__()
        self.fc1 = torch.nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.fc2 = torch.nn.Conv2d(channels // reduction, channels, kernel_size=1)
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        se = self.avg_pool(x)
        se = self.fc1(se)
        se = self.relu(se)
        se = self.fc2(se)
        se = self.sigmoid(se)
        return x * se

class EnsembleModelSE(torch.nn.Module):
    def __init__(self,
                 encoder_name,
                 encoder_weights,
                 in_channels,
                 active,
                 n_classes,
                 dropout_rate=0.1):
        super().__init__()

        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights
        self.in_channels = in_channels
        self.active = active
        self.n_classes = n_classes
        self.dropout_rate = dropout_rate

        if self.active == 'sigmoid':
            self.final_active = torch.nn.Sigmoid()
        elif self.active == 'softmax':
            self.final_active = torch.nn.Softmax(dim=1)
        elif self.active == 'linear':
            self.active = None
            self.final_active = torch.nn.Identity()
        else:
            self.active = None
            self.final_active = torch.nn.ReLU()

        self.model_b = smp.UnetPlusPlus(
            encoder_name=self.encoder_name,
            encoder_weights=self.encoder_weights,
            in_channels=int(self.in_channels / 2),
            classes=self.n_classes,
            activation=self.active
        )

        self.model_d = smp.UnetPlusPlus(
            encoder_name=self.encoder_name,
            encoder_weights=self.encoder_weights,
            in_channels=int(self.in_channels / 2),
            classes=self.n_classes,
            activation=self.active
        )

        # self.se_block_b = SqueezeExcitation(self.model_b.decoder.out_channels[-1], self.model_b.decoder.out_channels[-1] // 8)
        # self.se_block_d = SqueezeExcitation(self.model_d.decoder.out_channels[-1], self.model_b.decoder.out_channels[-1] // 8)
        # self.se_block_b = SqueezeExcitation(self.model_b.encoder.out_channels[-1], self.model_b.encoder.out_channels[-1] // 8)
        # self.se_block_d = SqueezeExcitation(self.model_d.encoder.out_channels[-1], self.model_d.encoder.out_channels[-1] // 8)
        # Define SE blocks for each stage
        print("self.model_b.encoder.out_channels:", self.model_b.encoder.out_channels)
        self.se_blocks_b = torch.nn.ModuleList([
            SqueezeExcitation(ch, ch // 8) for ch in self.model_b.encoder.out_channels            
        ])
        self.se_blocks_d = torch.nn.ModuleList([
            SqueezeExcitation(ch, ch // 8) for ch in self.model_d.encoder.out_channels
        ])

        self.segmentation_head = torch.nn.Sequential(
            torch.nn.LazyConv2d(64, kernel_size=1),#, padding=1),
            torch.nn.Dropout(p=self.dropout_rate),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, self.n_classes, kernel_size=1, bias=True)
        )

    def forward(self, input):
        b_input = input[:, 0:int(self.in_channels / 2), :, :]        
        d_input = input[:, int(self.in_channels / 2):self.in_channels, :, :]        
        
        # b_output = self.model_b.decoder(*self.model_b.encoder(b_input))
        # d_output = self.model_d.decoder(*self.model_d.encoder(d_input))
                
        # print("b shape:", b_output.shape)
        #  # Encoder outputs with SE blocks applied at each stage
        # b_encoded = self.model_b.encoder(b_input)
        # b_encoded_se = [self.se_blocks_b[i](b_encoded[i]) for i in range(len(b_encoded))]

        # d_encoded = self.model_d.encoder(d_input)
        # d_encoded_se = [self.se_blocks_d[i](d_encoded[i]) for i in range(len(d_encoded))]

        # # Encoder outputs
        # b_encoded = self.model_b.encoder(b_input)
        # d_encoded = self.model_d.encoder(d_input)
        
        # Apply SE blocks to the encoder output
        # if not isinstance(b_output, list):
        #     b_output = list(b_output)            
        # if not isinstance(d_output, list):
        #     d_output = list(d_output)
        
        # # apply SE only on encoder's last output
        # b_output[-1] = self.se_block_b(b_output[-1])
        # d_output[-1] = self.se_block_d(d_output[-1])
                
        # b_output = [b_out.unsqueeze(0) for b_out in b_output]        
        # b_output = torch.cat((b_output), dim=0)
        # d_output = [d_out.unsqueeze(0) for d_out in d_output]
        # d_output = torch.cat((d_output), dim=0)
        
        # forward inputs to encoders
        b_output = self.model_b.encoder(b_input)
        d_output = self.model_d.encoder(d_input)
        
        # apply SE on all encoder outputs        
        b_output = [se_block(x) for se_block, x in zip(self.se_blocks_b, b_output)]
        d_output = [se_block(x) for se_block, x in zip(self.se_blocks_d, d_output)]
                
        combined_output = torch.cat((b_output, d_output), dim=1)
        
        # concat without SE:
        # b_output = self.model_b.decoder(*self.model_b.encoder(b_input))
        # d_output = self.model_d.decoder(*self.model_d.encoder(d_input))
                
        # combined_output = torch.cat((b_output, d_output), dim=1)
                
        final_prediction = self.final_active(self.segmentation_head(combined_output))
        

        return final_prediction

        
class EnsembleModel(torch.nn.Module):
    def __init__(self,
                 encoder_name,
                encoder_weights,
                 in_channels,
                 active,
                 n_classes,
                 dropout_rate=0.1):
        super().__init__()

        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights
        self.in_channels = in_channels
        self.active = active
        self.n_classes = n_classes
        self.dropout_rate = dropout_rate 

        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # encoder = 'resnet34'
        # encoder_weights = 'imagenet'
                
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

        self.model_b = smp.UnetPlusPlus(
            encoder_name = self.encoder_name,
            encoder_weights = self.encoder_weights,
            in_channels = int(self.in_channels/2),
            classes = self.n_classes,
            activation = self.active
        )
        
        self.model_d = smp.UnetPlusPlus(
            encoder_name = self.encoder_name,
            encoder_weights = self.encoder_weights,
            in_channels = int(self.in_channels/2),
            classes = self.n_classes,
            activation = self.active
        )

        self.segmentation_head = torch.nn.Sequential(
            # HeInitLazyConv2d(64, kernel_size=1),
            torch.nn.LazyConv2d(64, kernel_size=1),
            torch.nn.Dropout(p=self.dropout_rate),
            #torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, self.n_classes, kernel_size=1, bias=True)
        )
        # Apply He initialization to convolutional layers
        # for layer in self.combine_layers:
        #     if isinstance(layer, torch.nn.Conv2d):
        #         init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')

    def forward(self,input):

        b_input = input[:,0:int(self.in_channels/2),:,:]
        d_input = input[:,int(self.in_channels/2):self.in_channels,:,:]
        b_output = self.model_b.decoder(*self.model_b.encoder(b_input))
        d_output = self.model_d.decoder(*self.model_d.encoder(d_input))

        combined_output = torch.cat((b_output,d_output),dim=1)
        final_prediction = self.final_active(self.segmentation_head(combined_output))
        
        return final_prediction

class EnsembleModelMIT(torch.nn.Module):
    def __init__(self,
                 encoder_name,
                encoder_weights,
                 in_channels,
                 active,
                 n_classes,
                 dropout_rate=0.1):
        super().__init__()

        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights
        self.in_channels = in_channels
        self.active = active
        self.n_classes = n_classes
        self.dropout_rate = dropout_rate 

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # encoder = 'mit_b5'
        # encoder_weights = 'imagenet'

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
            encoder_name = self.encoder_name,
            encoder_weights = self.encoder_weights,
            in_channels = int(self.in_channels/2),
            classes = self.n_classes,
            activation = self.active
        )
        
        self.model_d = smp.Unet(
            encoder_name = self.encoder_name,
            encoder_weights = self.encoder_weights,
            in_channels = int(self.in_channels/2),
            classes = self.n_classes,
            activation = self.active
        )

        self.segmentation_head = torch.nn.Sequential(
            HeInitLazyConv2d(64, kernel_size=1, bias=False),
            torch.nn.Dropout(p=self.dropout_rate),
            #torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, self.n_classes, kernel_size=1, bias=True)
        )
        # Apply He initialization to convolutional layers
        for layer in self.segmentation_head:
            if isinstance(layer, torch.nn.Conv2d):
                init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self,input):

        b_input = input[:,0:int(self.in_channels/2),:,:]
        d_input = input[:,int(self.in_channels/2):self.in_channels,:,:]
        b_output = self.model_b.decoder(*self.model_b.encoder(b_input))
        d_output = self.model_d.decoder(*self.model_d.encoder(d_input))

        self.features = torch.cat((b_output,d_output),dim=1)
        final_prediction = self.final_active(self.segmentation_head(self.features))
        
        return final_prediction

class EnsembleModelMAnet(torch.nn.Module):
    def __init__(self,
                 encoder_name,
                encoder_weights,
                 in_channels,
                 active,
                 n_classes,
                 dropout_rate=0.1):
        super().__init__()

        self.encoder_name = encoder_name    # "mit_b5"
        self.encoder_weights = encoder_weights  # "imagenet"
        self.decoder_channels=(1024, 512, 256, 128, 64)  # Decoder configuration
        self.in_channels = in_channels
        self.active = active
        self.n_classes = n_classes
        self.dropout_rate = dropout_rate 
        
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

        self.model_b = smp.MAnet(
            encoder_name = self.encoder_name,
            encoder_weights = self.encoder_weights,
            decoder_channels = self.decoder_channels,
            decoder_pab_channels = 256,  # Decoder Pyramid Attention Block channels
            in_channels = int(self.in_channels/2),
            classes = self.n_classes
        )
        
        self.model_d = smp.MAnet(
            encoder_name = self.encoder_name,
            encoder_weights = self.encoder_weights,
            decoder_channels = self.decoder_channels,
            decoder_pab_channels = 256,  # Decoder Pyramid Attention Block channels
            in_channels = int(self.in_channels/2),
            classes = self.n_classes
        )
        
        # Modify all activation functions in the encoder and decoder from ReLU to Mish
        _convert_activations(self.model_b.encoder, torch.nn.ReLU, torch.nn.Mish(inplace=True))
        _convert_activations(self.model_b.decoder, torch.nn.ReLU, torch.nn.Mish(inplace=True))
        
        _convert_activations(self.model_d.encoder, torch.nn.ReLU, torch.nn.Mish(inplace=True))
        _convert_activations(self.model_d.decoder, torch.nn.ReLU, torch.nn.Mish(inplace=True))       
        
        self.segmentation_head = DeepSegmentationHead(
            in_channels=self.decoder_channels[-2], out_channels=1
        )

        # self.segmentation_head = torch.nn.Sequential(
        #     HeInitLazyConv2d(64, kernel_size=1, bias=False),
        #     torch.nn.Dropout(p=self.dropout_rate),
        #     #torch.nn.BatchNorm2d(64),
        #     torch.nn.ReLU(inplace=True),
        #     torch.nn.Conv2d(64, self.n_classes, kernel_size=1, bias=True)
        # )
        # Apply He initialization to convolutional layers
        # for layer in self.segmentation_head:
        #     if isinstance(layer, torch.nn.Conv2d):
        #         init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self,input):

        b_input = input[:,0:int(self.in_channels/2),:,:]
        d_input = input[:,int(self.in_channels/2):self.in_channels,:,:]
        b_output = self.model_b.decoder(*self.model_b.encoder(b_input))
        d_output = self.model_d.decoder(*self.model_d.encoder(d_input))

        self.features = torch.cat((b_output,d_output),dim=1)
        final_prediction = self.final_active(self.segmentation_head(self.features))
        
        return final_prediction

class DeepSegmentationHead(torch.nn.Sequential):
    """Custom segmentation head for generating specific masks"""

    def __init__(
        self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1
    ):
        # Define a sequence of layers for the segmentation head
        layers = [
            torch.nn.Conv2d(
                in_channels,
                in_channels // 2,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            torch.nn.Mish(inplace=True),
            torch.nn.BatchNorm2d(in_channels // 2),
            torch.nn.Conv2d(
                in_channels // 2,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            torch.nn.UpsamplingBilinear2d(scale_factor=upsampling)
            if upsampling > 1
            else torch.nn.Identity(),
            Activation(activation) if activation else torch.nn.Identity(),
        ]
        super().__init__(*layers)


# class Discriminator(torch.nn.Module):
#     def __init__(self, img_size, channels):
#         super(Discriminator, self).__init__()
#         self.img_size = img_size
#         self.channels = channels
#         self.active = torch.nn.Sigmoid()

#         self.conv_blocks = torch.nn.Sequential(
#             torch.nn.Conv2d(channels, 64, kernel_size=4, stride=2, padding=1),
#             torch.nn.LeakyReLU(0.2, inplace=True),
#             torch.nn.Dropout(0.5),  # Add dropout layer with dropout probability of 0.5
#             torch.nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
#             torch.nn.BatchNorm2d(128),
#             torch.nn.LeakyReLU(0.2, inplace=True),
#         )        

#         self.fc = torch.nn.Sequential(
#             torch.nn.Linear(128 * (img_size[0] // 4) * (img_size[1] // 4), 512),
#             torch.nn.Dropout(0.5),  # Add dropout before the linear layer with dropout probability of 0.5
#             torch.nn.LeakyReLU(0.2, inplace=True),
#             torch.nn.Linear(512, 1)
#         )
        
#         # Initialize weights
#         self.apply(weights_init_normal)

#     def forward(self, source_imgs):
#         source_features = self.conv_blocks(source_imgs)
#         # target_features = self.conv_blocks(target_imgs)
        
#         source_features = source_features.view(source_features.size(0), -1)
#         # target_features = target_features.view(target_features.size(0), -1)
        
#         source_output = self.fc(source_features)        
#         source_output = self.active(source_output)
        
#         return source_output#, target_output


class Discriminator(torch.nn.Module):
  """
  Simple Discriminator network for domain adaptation with He initialization.
  """
  def __init__(self, in_channels, hidden_dim):
    super(Discriminator, self).__init__()
    # Define layers for feature extraction and classification
    self.layers = torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
        torch.nn.LeakyReLU(0.2),
        torch.nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
        torch.nn.LeakyReLU(0.2),
        torch.nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1)
    )

    # Apply He initialization to convolutional layers
    for layer in self.layers:
      if isinstance(layer, torch.nn.Conv2d):
        init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='leaky_relu')

  def forward(self, x):
    """
    Forward pass through the Discriminator network.

    Args:
        x: Input feature representation (tensor).

    Returns:
        Discriminator output (tensor).
    """
    return torch.sigmoid(self.layers(x))

class CrossAttention(torch.nn.Module):
    def __init__(self, in_channels):
        super(CrossAttention, self).__init__()
        self.scale = in_channels ** -0.5

    def forward(self, f_input, b_input):
        # Calculate attention scores
        attn = torch.matmul(f_input, b_input.transpose(-2, -1)) * self.scale
        
        # Apply softmax to get attention weights
        attn = torch.nn.functional.softmax(attn, dim=-1)
        
        # Attend to values (b_input) using attention weights
        attended_features = torch.matmul(attn, b_input)
        
        return attended_features
    
class AttentionModel(torch.nn.Module):
    def __init__(self, in_channels, active, n_classes):
        super().__init__()

        self.in_channels = in_channels
        self.active = active
        self.n_classes = n_classes

        encoder = 'mit_b5'
        encoder_weights = 'imagenet'

        if self.active == 'sigmoid':
            self.final_active = torch.nn.Sigmoid()
        elif self.active == 'softmax':
            self.final_active = torch.nn.Softmax(dim=1)
        elif self.active == 'linear':
            self.active = None
            self.final_active = torch.nn.Identity()
        else:
            self.active = None
            self.final_active = torch.nn.ReLU()

        self.model_b = smp.Unet(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=int(self.in_channels / 2),
            classes=self.n_classes,
            activation=self.active
        )

        self.model_d = smp.Unet(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=int(self.in_channels / 2),
            classes=self.n_classes,
            activation=self.active
        )

        self.attention = CrossAttention(in_channels=64)

        self.segmentation_head = torch.nn.Sequential(
            torch.nn.LazyConv2d(64, kernel_size=1),
            torch.nn.Dropout(p=0.1),
            # torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, self.n_classes, kernel_size=1)
        )

    def forward(self, input):
        b_input = input[:, 0:int(self.in_channels / 2), :, :]
        d_input = input[:, int(self.in_channels / 2):self.in_channels, :, :]
        b_output = self.model_b.decoder(*self.model_b.encoder(b_input))
        d_output = self.model_d.decoder(*self.model_d.encoder(d_input))

        # Apply cross-attention mechanism
        attended_b_output = self.cross_attention(d_output, b_output)

        # Concatenate the attended features
        combined_output = torch.cat((attended_b_output, d_output), dim=1)
        final_prediction = self.final_active(self.segmentation_head(combined_output))

        return final_prediction
    
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def _convert_activations(module, from_activation, to_activation):
    """Recursively convert activation functions in a module"""
    for name, child in module.named_children():
        if isinstance(child, from_activation):
            setattr(module, name, to_activation)
        else:
            _convert_activations(child, from_activation, to_activation)

class HeInitLazyConv2d(torch.nn.Module):
  """
  Wrapper module for LazyConv2d with He initialization.
  """
  def __init__(self, out_channels, kernel_size=1, bias=False):
    super(HeInitLazyConv2d, self).__init__()
    self.conv = torch.nn.LazyConv2d(out_channels, kernel_size, bias=bias)

  def forward(self, x):
    # He initialization on weight if not already done
    if not hasattr(self.conv, 'weight'):
      init.kaiming_normal_(self.conv.weight, mode='fan_in', nonlinearity='relu')
    return self.conv(x)

def create_model(model_details, n_classes):
    
    if model_details['architecture']=='Unet++':
        model =  smp.UnetPlusPlus(
            encoder_name = model_details['encoder'],
            encoder_weights = model_details['encoder_weights'],
            in_channels = model_details['in_channels'],
            classes = n_classes,
            activation = model_details['active']
            )
        
    elif model_details['architecture']=='ensemble':
        model = EnsembleModel(
            encoder_name = model_details['encoder'],
            encoder_weights = model_details['encoder_weights'],        
            in_channels = model_details['in_channels'],
            active = model_details['active'],
            n_classes = n_classes
            )
        
    elif model_details['architecture']=='ensembleMIT':
        model = EnsembleModelMIT(
            encoder_name = model_details['encoder'],
            encoder_weights = model_details['encoder_weights'],
            in_channels = model_details['in_channels'],
            active = model_details['active'],
            n_classes = n_classes
            )
    elif model_details['architecture']=='ensembleMAnet':
        model = EnsembleModelMAnet(
            encoder_name = model_details['encoder'],
            encoder_weights = model_details['encoder_weights'],
            in_channels = model_details['in_channels'],
            active = model_details['active'],
            n_classes = n_classes
        )
    
    elif model_details['architecture'] == 'ensembleSE':
        model = EnsembleModelSE(
            encoder_name = model_details['encoder'],
            encoder_weights = model_details['encoder_weights'],
            in_channels = model_details['in_channels'],
            active = model_details['active'],
            n_classes = n_classes
        )
    # if target:
    #     dummy_input = torch.zeros((2, 6, 512, 512))  # (BS, C, H, W)

    #     # Run a forward pass to initialize lazy parameters
    #     model(dummy_input)
    #     for param in model.parameters():
    #         param.detach_()
            
    
    return model