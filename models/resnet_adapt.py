import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, num_layers, in_channels, out_channels, identity_downsample=None, stride=1):
        assert num_layers in [18, 34, 50, 101, 152], "should be a a valid architecture"
        super(Block, self).__init__()

        self.num_layers = num_layers
        if self.num_layers > 34:
            self.expansion = 4
        else:
            self.expansion = 1

        # ResNet50, 101, and 152 include additional layer of 1x1 kernels
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm1d(out_channels)

        if self.num_layers > 34:
            self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        else:
            # for ResNet18 and 34, connect input directly to (3x3) kernel (skip first (1x1))
            self.conv2 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)

        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv3 = nn.Conv1d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm1d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x
        if self.num_layers > 34:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x

class ResNet(nn.Module):
    def __init__(self, num_layers, block, image_channels, target_time_dim, layer_norm, dim_change, feature_dim):
        assert num_layers in [18, 34, 50, 101, 152], f'ResNet{num_layers}: Unknown architecture! Number of layers has ' \
                                                     f'to be 18, 34, 50, 101, or 152 '
        super(ResNet, self).__init__()
        if num_layers < 50:
            self.expansion = 1
        else:
            self.expansion = 4
        if num_layers == 18:
            layers = [2, 2, 2, 2]
        elif num_layers == 34 or num_layers == 50:
            layers = [3, 4, 6, 3]
        elif num_layers == 101:
            layers = [3, 4, 23, 3]
        else:
            layers = [3, 8, 36, 3]
        self.in_channels = 64
        self.conv1 = nn.Conv1d(image_channels, 64, kernel_size=399, padding="same")
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=4, stride=2, padding=0)

        # ResNetLayers
        self.layer1 = self.make_layers(num_layers, block, layers[0], intermediate_channels=64, stride=1)
        self.layer2 = self.make_layers(num_layers, block, layers[1], intermediate_channels=128, stride=2)
        self.layer3 = self.make_layers(num_layers, block, layers[2], intermediate_channels=256, stride=2)
        self.layer4 = self.make_layers(num_layers, block, layers[3], intermediate_channels=512, stride=2)
        
        self.target_time_dim = target_time_dim
        self.avgpool = nn.AdaptiveAvgPool1d(self.target_time_dim)
        self.layer_norm = layer_norm
        self.dim_change = dim_change
        self.feature_dim = 512 * self.expansion
        if self.dim_change:
            self.feature_dim = feature_dim
            self.feat_change_dim = nn.Linear(512 * self.expansion, self.feature_dim)
            # self.feat_change_dim = nn.Sequential(
            #     nn.Conv1d(512 * self.expansion, self.feature_dim, 1, 1, padding="same"),
            #     nn.BatchNorm1d(self.feature_dim),
            #     nn.ReLU(),
            # )
        
        if self.layer_norm:
            self.feat_norm_layer = nn.LayerNorm(self.feature_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # print("ResNet before time dim change", x.shape)
        x = self.avgpool(x)
        
        x = x.permute(0, 2, 1)

        if self.dim_change:
            x = self.feat_change_dim(x)

        if self.layer_norm:
            x = self.feat_norm_layer(x)
        return x
        

    def make_layers(self, num_layers, block, num_residual_blocks, intermediate_channels, stride):
        layers = []

        identity_downsample = nn.Sequential(nn.Conv1d(self.in_channels, intermediate_channels*self.expansion, kernel_size=1, stride=stride),
                                            nn.BatchNorm1d(intermediate_channels*self.expansion))
        layers.append(block(num_layers, self.in_channels, intermediate_channels, identity_downsample, stride))
        self.in_channels = intermediate_channels * self.expansion # 256
        for i in range(num_residual_blocks - 1):
            layers.append(block(num_layers, self.in_channels, intermediate_channels)) # 256 -> 64, 64*4 (256) again
        return nn.Sequential(*layers)

def ResNet18(img_channels, target_time_dim, layer_norm, dim_change, feature_dim):
    return ResNet(18, Block, img_channels, target_time_dim, layer_norm, dim_change, feature_dim)


def ResNet34(img_channels, target_time_dim, layer_norm, dim_change, feature_dim):
    return ResNet(34, Block, img_channels, target_time_dim, layer_norm, dim_change, feature_dim)


def ResNet50(img_channels, target_time_dim, layer_norm, dim_change, feature_dim):
    return ResNet(50, Block, img_channels, target_time_dim, layer_norm, dim_change, feature_dim)


def ResNet101(img_channels, target_time_dim, layer_norm, dim_change, feature_dim):
    return ResNet(101, Block, img_channels, target_time_dim, layer_norm, dim_change, feature_dim)


def ResNet152(img_channels, target_time_dim, layer_norm, dim_change, feature_dim):
    return ResNet(152, Block, img_channels, target_time_dim, layer_norm, dim_change, feature_dim)

def ResNet_Adapt(num_layers: int = 34, channels: int = 1, time_dim: int = 112, layer_norm: bool = True, dim_change: bool = False, feat_dim: int = 128):
    # Dictionary mapping num_layers to the respective ResNet function
    resnet_functions = {
        18: ResNet18,
        34: ResNet34,
        50: ResNet50,
        101: ResNet101,
        152: ResNet152
    }

    # Check if the requested num_layers is available
    if num_layers in resnet_functions:
        # Call the appropriate ResNet function
        return resnet_functions[num_layers](img_channels=channels, target_time_dim=time_dim, layer_norm=layer_norm, dim_change=dim_change, feature_dim=feat_dim)
    else:
        raise ValueError(f"ResNet for {num_layers} layers is not defined.")

