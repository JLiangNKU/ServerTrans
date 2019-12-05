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
            # nn.InstanceNorm2d(in_features),
            nn.LeakyReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            # nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)

class GeneratorResNet_original(nn.Module):
    def __init__(self, input_shape, num_residual_blocks):
        super(GeneratorResNet_original, self).__init__()

        channels = input_shape[0]

        # Initial convolution block
        out_features = 64
        model = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, out_features, 7),
            # nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features

        # Downsampling
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                # nn.InstanceNorm2d(out_features),
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
                # nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Output layer
        model += [nn.ReflectionPad2d(channels), nn.Conv2d(out_features, channels, 7), nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return x + self.model(x)


class ResidualBlock_high_original(nn.Module):
    def __init__(self, input_shape, num_residual_blocks, levels):
        super(ResidualBlock_high_original, self).__init__()

        channels = input_shape[0]

        # Initial convolution block
        out_features = 64
        model = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, out_features, 7),
            # nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features

        # Downsampling
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                # nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Residual blocks
        for _ in range(5):
            model += [ResidualBlock(out_features)]

        # Upsampling
        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                # nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Output layer
        model += [nn.ReflectionPad2d(channels), nn.Conv2d(out_features, 1, 7), nn.Tanh()]

        self.model = nn.Sequential(*model)

        self.trans_mask_block_1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 16, 3),
            # nn.InstanceNorm2d(16),
            nn.LeakyReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(16, 1, 3),
            nn.Tanh()
        )
        self.trans_mask_block_2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 16, 3),
            # nn.InstanceNorm2d(16),
            nn.LeakyReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(16, 1, 3),
            nn.Tanh()
        )
        self.trans_mask_block_3 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 16, 3),
            # nn.InstanceNorm2d(16),
            nn.LeakyReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(16, 1, 3),
            nn.Tanh()
        )

        self.trans_mask_blocks = []
        for _ in range(levels):
            trans_mask_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 16, 3),
            # nn.InstanceNorm2d(16),
            nn.LeakyReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(16, 1, 3),
            nn.Tanh())
            self.trans_mask_blocks.append(trans_mask_block)


    def forward(self, x, if_conv, levels):

        mask = self.model(x)

        masks = []
        masks.append(mask)

        for i in range(levels):

            mask = nn.functional.upsample_bilinear(mask, scale_factor=2)
            mask = self.trans_mask_blocks[i](mask)
            masks.append(mask)

        return masks

##############################
#        Discriminator
##############################

class MultiDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(MultiDiscriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # Extracts three discriminator models
        self.models = nn.ModuleList()
        for i in range(3):
            self.models.add_module(
                "disc_%d" % i,
                nn.Sequential(
                    *discriminator_block(in_channels, 64, normalize=False),
                    *discriminator_block(64, 128),
                    *discriminator_block(128, 256),
                    *discriminator_block(256, 512),
                    nn.Conv2d(512, 1, 3, padding=1)
                ),
            )

        self.downsample = nn.AvgPool2d(in_channels, stride=2, padding=[1, 1], count_include_pad=False)

    def compute_loss(self, x, gt):
        """Computes the MSE between model output and scalar gt"""
        loss = sum([torch.mean((out - gt) ** 2) for out in self.forward(x)])
        return loss

    def forward(self, x):
        outputs = []
        for m in self.models:
            outputs.append(m(x))
            x = self.downsample(x)
        return outputs
