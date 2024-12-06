import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, patch_size, stride, padding, activation):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, patch_size, stride, padding),
            nn.BatchNorm2d(out_channel),
            activation
        )
    
    def forward(self, x):
        x = self.conv(x)
        return x

class LayerBlock(nn.Module):
    def __init__(self, conv1, conv2):
        super(LayerBlock, self).__init__()
        self.conv1 = conv1
        self.conv2 = conv2
        
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        return x + residual
    
class ResNet_My(nn.Module):
    def __init__(self, input_size, dim, block_num_1, block_num_2):
        super(ResNet_My, self).__init__()
        self.pool = nn.MaxPool2d(2)
        self.first_conv = ConvBlock(3, 64, 7, 2, 3, nn.ReLU())

        self.layer1 = self._make_layer(64, 64*dim, 3, 1, 1, 1, nn.ReLU(), block_num_1)
        self.layer2 = self._make_layer(64*dim*2, 64*dim*2, 3, 1, 1, 1, nn.ReLU(), block_num_2)
        self.layer3 = self._make_layer(64*dim*2 ** 1, 64*dim*2 ** 1, 3, 1, 1, 1, nn.ReLU(), block_num_2)
        self.layer4 = self._make_layer(64*dim*2 ** 2, 64*dim*2 ** 2, 3, 1, 1, 1, nn.ReLU(), block_num_2)

        self.avglayer = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(2304, 10),
        )
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(64*dim, 64*dim*2, 3, 2, 1, 1),
            nn.BatchNorm2d(64*dim*2),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64*dim*2, 64*dim*2, 3, 2, 1, 1),
            nn.BatchNorm2d(64*dim*2),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64*dim*2, 64*dim*2*2, 3, 2, 1, 1),
            nn.BatchNorm2d(64*dim*2*2),
            nn.ReLU()
        )

        
    def _make_layer(self, in_channel, out_channel, patch_size, stride1 ,stride2, padding, activation, block_num):
        layers = []
        for _ in range(block_num):
            layers.append(self.create_block(in_channel, out_channel, patch_size, stride1, stride2, padding, activation))
        return nn.Sequential(*layers)
    
    def create_block(self, in_channel, out_channel, patch_size, stride1 ,stride2, padding, activation):
        return nn.Sequential(
            LayerBlock(ConvBlock(in_channel, out_channel, patch_size, stride1, padding, activation), 
                       ConvBlock(out_channel, out_channel, patch_size, stride2, padding, activation))
        )
        
    def forward(self, x):
        x = self.first_conv(x)
        x = self.pool(x)
        x = self.layer1(x)
        x = self.conv1(x)
        x = self.layer2(x)
        x = self.conv2(x)
        # print(x.shape)
        x = self.layer3(x)
        x = self.conv3(x)
        x = self.layer4(x)
        x = self.avglayer(x)
        # print(x.shape)
        x = self.fc(x)
        return x
        