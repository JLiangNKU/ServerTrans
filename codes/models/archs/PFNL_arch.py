import torch
import torch.nn as nn
import torch.nn.functional as F
import models.archs.arch_util as arch_util


class PixelUnshuffle(nn.Module):
    '''Pixel Unshuffle Layer'''
    def __init__(self, downscale_factor):
        super(PixelUnshuffle, self).__init__()
        self.downscale_factor = downscale_factor

    def forward(self, x):
        '''
        :param x: (B, C, r*H, r*W)
        :return: (B, r*r*C, H, W)
        '''
        b, c, h, w = x.shape
        oc = c * self.downscale_factor * self.downscale_factor
        oh = h // self.downscale_factor
        ow = w // self.downscale_factor
        x = x.reshape(b, c, oh, self.downscale_factor, ow, self.downscale_factor)
        return x.permute(0, 1, 3, 5, 2, 4).reshape(b, oc, oh, ow)


class NonLocalResBlock(nn.Module):

    def __init__(self, nc=3, nt=5, r=2):
        super(NonLocalResBlock, self).__init__()
        self.space_to_depth = PixelUnshuffle(r)
        self.depth_to_space = nn.PixelShuffle(r)
        self.g = nn.Conv2d(nc * nt * r * r, nc * nt * r * r, kernel_size=1, stride=1, padding=0, bias=True)
        self.w = nn.Conv2d(nc * nt * r * r, nc * nt * r * r, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        '''
        :param x: (B, T, C, H, W)
        :return: (B, T, C, H, W)
        '''
        B, T, C, H, W = x.shape
        # x1: (B, T*C, H, W)
        x1 = torch.reshape(x, (B, T*C, H, W))
        # x2: (B, T*C*r*r, H/r, W/r)
        x2 = self.space_to_depth(x1)

        # x3: (B, H/r*W/r, T*C*r*r)
        x3 = torch.reshape(x2, (x2.shape[0], x2.shape[1], x2.shape[2] * x2.shape[3])).permute(0, 2, 1)
        # x4: (B, T*C*r*r, H/r*W/r)
        x4 = torch.reshape(x2, (x2.shape[0], x2.shape[1], x2.shape[2] * x2.shape[3])).permute(0, 1, 2)
        # F: (B, H/r*W/r, H/r*W/r)
        F = torch.bmm(x3, x4)
        F = torch.softmax(F, dim=-1)
        # G: (B, H/r*W/r, T*C*r*r)
        G = torch.reshape(self.g(x2).permute(0, 2, 3, 1), (x2.shape[0], x2.shape[2] * x2.shape[3], x2.shape[1]))
        # Y: (B, H/r*W/r, T*C*r*r)
        Y = torch.bmm(F, G)
        # Y: (B, T*C*r*r, H/r, W/r)
        Y = torch.reshape(Y.permute(0, 2, 1), (x2.shape[0], x2.shape[1], x2.shape[2], x2.shape[3]))

        # Z: (B, T, C, H, W)
        Z = self.depth_to_space(self.w(Y))
        Z = torch.reshape(Z, (B, T, C, H, W)) + x

        return Z


class FusionResBlock(nn.Module):

    def __init__(self, nf=64, nt=5):
        super(FusionResBlock, self).__init__()
        self.conv_1 = nn.Conv2d(nf * 1 , nf * 1, 3, stride=1, padding=1, dilation=1, bias=True)
        self.conv_2 = nn.Conv2d(nf * nt, nf * 1, 1, stride=1, padding=0, dilation=1, bias=True)
        self.conv_3 = nn.Conv2d(nf * 2 , nf * 1, 3, stride=1, padding=1, dilation=1, bias=True)


    def forward(self, x):
        '''
        :param x: (B, T, C, H, W)
        :return: (B, T*C, H, W)
        '''
        B, T, C, H, W = x.shape
        I0_list = torch.chunk(x, T, dim=1)
        I1_list = []
        for t in range(T):
            I1_list.append(self.conv_1(I0_list[t].squeeze(1)))
        I1 = torch.cat(I1_list, dim=1)
        I2 = self.conv_2(I1)
        O_list = []
        for t in range(T):
            O_list.append(self.conv_3(torch.cat([I1_list[t], I2], dim=1)))
            O_list[t] += I0_list[t].squeeze(1)
        out = torch.cat(O_list, dim=1)
        return out


class PFNL(nn.Module):

    def __init__(self, nf=64, nc=3, nt=5, r=2, scale=4):
        super(PFNL, self).__init__()
        self.scale = scale
        self.nl_block = NonLocalResBlock(nc=nc, nt=nt, r=r)
        self.pf_block = FusionResBlock(nf=nf, nt=nt)
        self.conv_inp = nn.Conv2d(nc * 1 , nf * 1, 5, stride=1, padding=2, dilation=1, bias=True)
        self.conv_mer = nn.Conv2d(nf * nt, nf * 1, 1, stride=1, padding=0, dilation=1, bias=True)
        self.upsample = arch_util.Upsampler(arch_util.default_conv, scale, nf)
        self.conv_out = nn.Conv2d(nf, 3, 3, stride=1, padding=1, dilation=1, bias=True)

    def forward(self, x):
        '''
        :param x: (B, T, C, H, W)
        :return: (B, T*C, H, W)
        '''
        B, T, C, H, W = x.shape
        bic = F.interpolate(x[:, T//2, :, :, :], scale_factor=self.scale, mode='bicubic')
        x = self.nl_block(x)
        x_list = []
        for t in range(T):
            x_list.append(self.conv_inp(x[:, t, :, :, :]))
        x = torch.stack(x_list, dim=1)
        x = self.pf_block(x)
        x = self.conv_mer(x)
        x = self.upsample(x)
        x = self.conv_out(x) + bic
        return x


if __name__ == '__main__':
    # B, T, C, H, W = 4, 5, 3, 128, 128
    # x = torch.randn(B, T, C, H, W)
    # NL = NonLocalResBlock(r=2)
    # o1 = NL(x)
    # print(o1.shape)
    # B, T, C, H, W = 4, 5, 64, 128, 128
    # x = torch.randn(B, T, C, H, W)
    # PF = FusionResBlock()
    # o2 = PF(x)
    # print(o2.shape)
    B, T, C, H, W = 4, 5, 3, 128, 128
    x = torch.randn(B, T, C, H, W)
    PFNL = PFNL(nf=64, nc=3, nt=5, r=2, scale=4)
    o3 = PFNL(x)
    print(o3.shape)