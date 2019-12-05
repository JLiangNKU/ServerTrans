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


class GeneratorResNet(nn.Module):
    def __init__(self, input_shape, num_residual_blocks):
        super(GeneratorResNet, self).__init__()

        # Initial convolution block
        model = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, 16, 3),
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(),
        ]
        model += [
            nn.ReflectionPad2d(1),
            nn.Conv2d(16, 64, 3),
            # nn.InstanceNorm2d(64),
            nn.LeakyReLU(),
        ]

        # Residual blocks
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(64)]

        # Output layer
        model += [
            nn.ReflectionPad2d(1), 
            nn.Conv2d(64, 16, 3), 
            # nn.InstanceNorm2d(32),
            nn.LeakyReLU(),
            ]
        model += [
            nn.ReflectionPad2d(1), 
            nn.Conv2d(16, 3, 3), 
            ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = x + self.model(x)
        out = nn.functional.tanh(out)
        return out


class ResidualBlock_high(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock_high, self).__init__()

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

class Trans_high(nn.Module):
    def __init__(self):
        super(Trans_high, self).__init__()

        model_trans = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, 64, 3),
            # nn.InstanceNorm2d(16),
            nn.LeakyReLU(),
        ]

        # Residual blocks
        for _ in range(2):
            model_trans += [ResidualBlock_high(64)]

        # Output layer
        model_trans += [
            nn.ReflectionPad2d(1), 
            nn.Conv2d(64, 1, 3), 
            # nn.Tanh()
            ]

        self.model = nn.Sequential(*model_trans)

        self.trans_mask_block_1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 16, 3),
            # nn.InstanceNorm2d(16),
            nn.LeakyReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(16, 1, 3),
            # nn.Tanh()
        )
        self.trans_mask_block_2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 16, 3),
            # nn.InstanceNorm2d(16),
            nn.LeakyReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(16, 1, 3),
            # nn.Tanh()
        )
        self.trans_mask_block_3 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 16, 3),
            # nn.InstanceNorm2d(16),
            nn.LeakyReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(16, 1, 3),
            # nn.Tanh()
        )

    def forward(self, x, if_conv):

        mask_0 = self.model(x)

        mask_1 = nn.functional.upsample_bilinear(mask_0, scale_factor=2)
        if if_conv:
            mask_1 = self.trans_mask_block_1(mask_1)

        mask_2 = nn.functional.upsample_bilinear(mask_1, scale_factor=2)
        if if_conv:
            mask_2 = self.trans_mask_block_2(mask_2)

        # mask_3 = nn.functional.upsample_bilinear(mask_2, scale_factor=2)
        # if if_conv:
        #     mask_3 = self.trans_mask_block_3(mask_3)

        return mask_0, mask_1, mask_2#, mask_3

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
            nn.LeakyReLU(),
        ]
        in_features = out_features

        # Downsampling
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                # nn.InstanceNorm2d(out_features),
                nn.LeakyReLU(),
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
                nn.LeakyReLU(),
            ]
            in_features = out_features

        # Output layer
        model += [nn.ReflectionPad2d(channels), nn.Conv2d(out_features, channels, 7)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = x + self.model(x)
        out = nn.functional.tanh(out)
        return out 

class ResidualBlock_high_original(nn.Module):
    def __init__(self, input_shape, num_residual_blocks):
        super(ResidualBlock_high_original, self).__init__()

        channels = 3

        # Initial convolution block
        out_features = 64
        model = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(9, out_features, 7),
            # nn.InstanceNorm2d(out_features),
            nn.LeakyReLU(),
        ]
        in_features = out_features

        # Downsampling
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                # nn.InstanceNorm2d(out_features),
                nn.LeakyReLU(),
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
                nn.LeakyReLU(),
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
