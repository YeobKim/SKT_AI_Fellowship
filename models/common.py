import torch
import torch.nn as nn

def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)


def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    # print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h

class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)

class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)

class RCA_Block(nn.Module):
    def __init__(self, features):
        super(RCA_Block, self).__init__()
        firstblock = []
        firstblock.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1))
        firstblock.append(nn.PReLU())
        firstblock.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1))
        self.firstblock = nn.Sequential(*firstblock)

        ch_attention = []
        ch_attention.append(nn.AdaptiveAvgPool2d(1))
        ch_attention.append(nn.Conv2d(in_channels=features, out_channels=features // 16, kernel_size=1, padding=0))
        ch_attention.append(nn.PReLU())
        ch_attention.append(nn.Conv2d(in_channels=features // 16, out_channels=features, kernel_size=1, padding=0))
        ch_attention.append(nn.Sigmoid())
        self.cab = nn.Sequential(*ch_attention)

    def forward(self, x):
        residual = x
        data = self.firstblock(x)
        ch_data = self.cab(data) * data
        out = ch_data + residual

        return out

class ARCA_Block(nn.Module):
    def __init__(self, features):
        super(ARCA_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=1, padding=0)
        self.conv6 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=6, dilation=6)
        self.conv12 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=12, dilation=12)
        self.conv18 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=18, dilation=18)

        self.RCAB = RCA_Block(features)
        platlayer = []
        platlayer.append(nn.PReLU())
        platlayer.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=1, padding=0))
        self.platlayer = nn.Sequential(*platlayer)

        self.feat2feat = nn.Conv2d(in_channels=features * 4, out_channels=features, kernel_size=1, padding=0)

    def forward(self, x):
        # ASPP Block
        d1 = self.conv1(x)
        d6 = self.conv6(x)
        d12 = self.conv12(x)
        d18 = self.conv18(x)
        dcat = torch.cat((d1, d6, d12, d18), 1)
        dilatedcat = self.feat2feat(dcat)
        aspp = self.platlayer(dilatedcat)
        out = self.RCAB(aspp)

        return out

class ChannelAttention(nn.Module):
    def __init__(self, features):
        super(ChannelAttention, self).__init__()
        ch_attention = []
        ch_attention.append(nn.AdaptiveAvgPool2d(1))
        ch_attention.append(nn.Conv2d(in_channels=features, out_channels=features // 16, kernel_size=1, padding=0))
        ch_attention.append(nn.PReLU())
        ch_attention.append(nn.Conv2d(in_channels=features // 16, out_channels=features, kernel_size=1, padding=0))
        ch_attention.append(nn.Sigmoid())
        self.cab = nn.Sequential(*ch_attention)

    def forward(self, x):
        out = self.cab(x) * x

        return out

class ResidudalGroup(nn.Module):
    def __init__(self, features):
        super(ResidudalGroup, self).__init__()

        block1 = []
        block1.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1))
        block1.append(nn.PReLU())
        block1.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1))

        self.block1 = nn.Sequential(*block1)

        self.basicconv = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x

        head = self.block1(x) + x
        body = self.block1(head) + head
        tail = self.block1(body) + body

        out = self.basicconv(tail) + residual

        return out

class make_dense(nn.Module):
      def __init__(self, nChannels, growthRate, kernel_size=3):
        super(make_dense, self).__init__()
        self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size-1)//2)
        self.prelu = nn.PReLU()
      def forward(self, x):
        out = self.prelu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out

# Residual dense block (RDB) architecture
class RDB(nn.Module):
      def __init__(self, features, nDenselayer):
        super(RDB, self).__init__()
        nChannels_ = features
        growthRate = features // 2
        modules = []
        for i in range(nDenselayer):
            modules.append(make_dense(nChannels_, growthRate))
            nChannels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(nChannels_, features, kernel_size=1, padding=0)
      def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out)
        out = out + x
        return out

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SCAB(nn.Module):
    def __init__(self, features):
        super(SCAB, self).__init__()
        firstblock = []
        firstblock.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1))
        firstblock.append(nn.PReLU())
        firstblock.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1))
        self.firstblock = nn.Sequential(*firstblock)

        ch_attention = []
        ch_attention.append(nn.AdaptiveAvgPool2d(1))
        ch_attention.append(nn.Conv2d(in_channels=features, out_channels=features // 16, kernel_size=1, padding=0))
        ch_attention.append(nn.PReLU())
        ch_attention.append(nn.Conv2d(in_channels=features // 16, out_channels=features, kernel_size=1, padding=0))
        ch_attention.append(nn.Sigmoid())
        self.cab = nn.Sequential(*ch_attention)

        self.compress = ChannelPool()

        sa_attention = []
        sa_attention.append(nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, padding=0))
        sa_attention.append(nn.PReLU())
        sa_attention.append(nn.Sigmoid())
        self.sab = nn.Sequential(*sa_attention)

    def forward(self, x):
        residual = x
        data = self.firstblock(x)
        sp_data = self.compress(data)
        ch_data = self.cab(data) * data
        output = self.sab(sp_data) * ch_data
        out = output + residual

        return out

class SAB(nn.Module):
    def __init__(self, nchannel):
        super(SAB, self).__init__()
        sa_attention = []
        sa_attention.append(nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, padding=0))
        sa_attention.append(nn.PReLU())
        sa_attention.append(nn.Sigmoid())
        self.sab = nn.Sequential(*sa_attention)

        self.compress = ChannelPool()

    def forward(self, x):
        out = self.sab(self.compress(x)) * x

        return out


class aspp_feat(nn.Module):
    def __init__(self, channels, features):
        super(aspp_feat, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=3, padding=1)

        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=3, padding=2, dilation=2)

        self.conv4 = nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=3, padding=4, dilation=4)

        self.conv8 = nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=3, padding=8, dilation=8)

        platlayer = []
        platlayer.append(nn.PReLU())
        platlayer.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1))
        self.platlayer = nn.Sequential(*platlayer)

        self.prelu = nn.PReLU()
        self.feat2feat = nn.Conv2d(in_channels=features * 4, out_channels=features, kernel_size=1, padding=0)
        self.feat2ch = nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=1, padding=0)

    def forward(self, x):
        d1 = self.conv1(x)
        d2 = self.conv2(x)
        d4 = self.conv4(x)
        d8 = self.conv8(x)
        dcat = self.feat2feat(torch.cat([d1, d2, d4, d8], 1))

        out = self.platlayer(dcat)

        return out

class sksak_feat(nn.Module):
    def __init__(self, channels, features):
        super(sksak_feat, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=3, padding=1)

        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=3, padding=2, dilation=2)

        self.conv4 = nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=3, padding=4, dilation=4)

        self.conv8 = nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=3, padding=8, dilation=8)

        platlayer = []
        platlayer.append(nn.PReLU())
        platlayer.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1))
        self.platlayer = nn.Sequential(*platlayer)

        self.prelu = nn.PReLU()
        self.feat2feat = nn.Conv2d(in_channels=features * 4, out_channels=features, kernel_size=1, padding=0)
        self.feat2ch = nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=1, padding=0)
        self.ch4feat = nn.Conv2d(in_channels=channels * 4, out_channels=features, kernel_size=1, padding=0)

    def forward(self, x, edge):
        d1 = self.conv1(x)
        d2 = self.conv2(x)
        d4 = self.conv4(x)
        d8 = self.conv8(x)
        dcat = self.feat2feat(torch.cat([d1, d2, d4, d8], 1))

        dout = self.feat2ch(self.platlayer(dcat))

        out = self.ch4feat(torch.cat([x, edge, x+edge, dout], 1))

        return out

class video_feat(nn.Module):
    def __init__(self, channels, features):
        super(video_feat, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=3, padding=2, dilation=2)
        self.conv4 = nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=3, padding=4, dilation=4)
        self.conv8 = nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=3, padding=8, dilation=8)

        platlayer = []
        platlayer.append(nn.PReLU())
        platlayer.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1))
        self.platlayer = nn.Sequential(*platlayer)

        self.prelu = nn.PReLU()
        self.feat2feat = nn.Conv2d(in_channels=features * 4, out_channels=features, kernel_size=1, padding=0)
        self.feat2ch = nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=1, padding=0)
        self.ch4feat = nn.Conv2d(in_channels=channels * 4, out_channels=features, kernel_size=1, padding=0)

    def forward(self, x, edge):
        d1 = self.conv1(x)
        d2 = self.conv2(x)
        d4 = self.conv4(x)
        d8 = self.conv8(x)
        dcat = self.feat2feat(torch.cat([d1, d2, d4, d8], 1))

        dout = self.feat2ch(self.platlayer(dcat))

        out = self.ch4feat(torch.cat([x, dout], 1))

        return out


class ResidudalBlock(nn.Module):
    def __init__(self, features):
        super(ResidudalBlock, self).__init__()

        block1 = []
        block1.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1))
        block1.append(nn.PReLU())
        block1.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1))

        self.block1 = nn.Sequential(*block1)

    def forward(self, x):
        residual = x

        out = self.block1(x) + residual

        return out

class sksak_feat2(nn.Module):
    def __init__(self, channels, features):
        super(sksak_feat2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=features//4, kernel_size=3, padding=1)

        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=features//4, kernel_size=3, padding=2, dilation=2)

        self.conv4 = nn.Conv2d(in_channels=channels, out_channels=features//4, kernel_size=3, padding=4, dilation=4)

        self.conv8 = nn.Conv2d(in_channels=channels, out_channels=features//4, kernel_size=3, padding=8, dilation=8)

        platlayer = []
        platlayer.append(nn.PReLU())
        platlayer.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1))
        self.platlayer = nn.Sequential(*platlayer)

        self.prelu = nn.PReLU()
        self.feat2feat = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=1, padding=0)
        self.feat2ch = nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=1, padding=0)
        self.ch4feat = nn.Conv2d(in_channels=channels * 4, out_channels=features, kernel_size=1, padding=0)

    def forward(self, x, edge):
        d1 = self.conv1(x)
        d2 = self.conv2(x)
        d4 = self.conv4(x)
        d8 = self.conv8(x)
        dcat = torch.cat([d1, d2, d4, d8], 1)

        dout = self.feat2ch(self.platlayer(dcat))

        out = self.ch4feat(torch.cat([x, edge, x+edge, dout], 1))

        return out

class sksak_edgecomb(nn.Module):
    def __init__(self, channels, features):
        super(sksak_edgecomb, self).__init__()

        platlayer = []
        platlayer.append(nn.PReLU())
        platlayer.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1))
        self.platlayer = nn.Sequential(*platlayer)

        self.prelu = nn.PReLU()
        self.feat2feat = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=1, padding=0)
        self.feat2ch = nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=1, padding=0)
        self.ch3feat = nn.Conv2d(in_channels=channels * 3, out_channels=features, kernel_size=1, padding=0)

    def forward(self, x, edge):
        out = self.ch3feat(torch.cat([x, edge, x+edge], 1))

        return out
