import torch.nn as nn

class FullyConvNetwork(nn.Module):

    def __init__(self):
        super().__init__()
         # Encoder (Convolutional Layers)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=4, stride=2, padding=1),  # Input channels: 3, Output channels: 8
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )
        ### FILL: add more CONV Layers
        # Conv1
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=100)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv2
        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv3
        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv4
        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv5
        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers as convolutional layers
        self.fc6_cs = nn.Conv2d(in_channels=512, out_channels=4096, kernel_size=7, stride=1, padding=0)
        self.relu6_cs = nn.ReLU(inplace=True)
        self.fc7_cs = nn.Conv2d(in_channels=4096, out_channels=4096, kernel_size=1, stride=1, padding=0)
        self.relu7_cs = nn.ReLU(inplace=True)
        
        # Final score layer
        self.score_fr = nn.Conv2d(in_channels=4096, out_channels=3, kernel_size=1, stride=1, padding=0)
        
        # Decoder (Deconvolutional Layers)
        ### FILL: add ConvTranspose Layers
        ### None: since last layer outputs RGB channels, may need specific activation function

        # Upsampling
        self.upscore2 = nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=4, stride=2, bias=False)
        
        # Additional layers
        self.score_pool4 = nn.Conv2d(in_channels=512, out_channels=3, kernel_size=1, stride=1, padding=0)
        self.upscore_pool4 = nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=4, stride=2, bias=False)
        self.score_pool3 = nn.Conv2d(in_channels=256, out_channels=3, kernel_size=1, stride=1, padding=0)
        self.upscore8 = nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=16, stride=8, bias=False)
        self.tanh_upscore8= nn.Tanh()

        
    def forward(self, x):
        # Encoder forward pass
        
        # Decoder forward pass
        
        ### FILL: encoder-decoder forward pass
        
        # Save the input data for later use in the crop operation
        input_data = x
        
        # Conv1
        x = self.relu1_1(self.conv1_1(x))
        x = self.relu1_2(self.conv1_2(x))
        x = self.pool1(x)
        
        # Conv2
        x = self.relu2_1(self.conv2_1(x))
        x = self.relu2_2(self.conv2_2(x))
        x = self.pool2(x)
        
        # Conv3
        x = self.relu3_1(self.conv3_1(x))
        x = self.relu3_2(self.conv3_2(x))
        x = self.relu3_3(self.conv3_3(x))
        pool3 = self.pool3(x)  # Save the output of pool3 for later use
        
        # Conv4
        x = self.relu4_1(self.conv4_1(pool3))
        x = self.relu4_2(self.conv4_2(x))
        x = self.relu4_3(self.conv4_3(x))
        pool4 = self.pool4(x)  # Save the output of pool4 for later use
        
        # Conv5
        x = self.relu5_1(self.conv5_1(pool4))
        x = self.relu5_2(self.conv5_2(x))
        x = self.relu5_3(self.conv5_3(x))
        x = self.pool5(x)
        
        # Fully connected layers as convolutional layers
        x = self.relu6_cs(self.fc6_cs(x))
        x = self.relu7_cs(self.fc7_cs(x))
        
        # Final score layer
        x = self.score_fr(x)
        
        # Upsampling
        upscore2 = self.upscore2(x)
        
        # Score pool4
        score_pool4 = self.score_pool4(pool4)
        score_pool4c = self.crop(score_pool4, upscore2, 5)
        
        # Fuse pool4
        fuse_pool4 = upscore2 + score_pool4c
        upscore_pool4 = self.upscore_pool4(fuse_pool4)
        
        # Score pool3
        score_pool3 = self.score_pool3(pool3)
        score_pool3c = self.crop(score_pool3, upscore_pool4, 9)
        
        # Fuse pool3
        fuse_pool3 = upscore_pool4 + score_pool3c
        upscore8 = self.upscore8(fuse_pool3)
        
        # Tanh activation
        upscore8 = self.tanh_upscore8(upscore8)
        
        # Crop the upscore8 to match the size of the input data
        score = self.crop(upscore8, input_data, 31)
        
        output = score
        
        return output
    
    def crop(self, x, target, offset):
        _, _, h, w = target.size()
        return x[:, :, offset:offset + h, offset:offset + w]
    
    