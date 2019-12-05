import torch.nn as nn
import torch.nn.functional as F
import torch


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


##############################
#           RESNET
##############################


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)


class GeneratorResNet(nn.Module):
    def __init__(self, input_shape, num_residual_blocks):
        super(GeneratorResNet, self).__init__()

        channels = input_shape[0]

        # Initial convolution block
        out_features = 64
        model = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features

        # Downsampling
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Residual blocks
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]

        # Upsampling
        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Output layer
        model += [nn.ReflectionPad2d(channels), nn.Conv2d(out_features, channels, 7), nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class trans_mask_block(nn.Module):
    def __init__(self):
        super(trans_mask_block, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 64, 3),
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 1, 3),
            nn.Tanh()
        )

    def forward(self, x):
        return self.block(x)

class Trans_high(nn.Module):
    def __init__(self):
        super(Trans_high, self).__init__()

        model_trans = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(9, 64, 3),
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(),
        ]

        for _ in range(3):
            model_trans += [ResidualBlock(64)]

        model_trans += [
            nn.ReflectionPad2d(1), 
            nn.Conv2d(64, 1, 3), 
            nn.Tanh()
            ]

        self.model = nn.Sequential(*model_trans)

        self.trans_mask_block_1 = trans_mask_block()
        self.trans_mask_block_2 = trans_mask_block()
        self.trans_mask_block_3 = trans_mask_block()

    def forward(self, x, if_conv):

        mask_0 = self.model(x)

        mask_1 = nn.functional.upsample_bilinear(mask_0, scale_factor=2)
        if if_conv:
            mask_1 = self.trans_mask_block_1(mask_1)

        mask_2 = nn.functional.upsample_bilinear(mask_1, scale_factor=2)
        if if_conv:
            mask_2 = self.trans_mask_block_2(mask_2)

        mask_3 = nn.functional.upsample_bilinear(mask_2, scale_factor=2)
        if if_conv:
            mask_3 = self.trans_mask_block_3(mask_3)

        return mask_0, mask_1, mask_2, mask_3


##############################
#        Discriminator
##############################


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        channels, height, width = input_shape

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)
