import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import models.common as common
import torchvision.ops.deform_conv as dc


##########################################################################
def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


##########################################################################
## Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.PReLU(),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


##########################################################################
## Channel Attention Block (CAB)
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        firstblock = []
        firstblock.append(nn.Conv2d(in_channels=n_feat, out_channels=n_feat, kernel_size=3, padding=1, bias=False))
        firstblock.append(nn.PReLU())
        firstblock.append(nn.Conv2d(in_channels=n_feat, out_channels=n_feat, kernel_size=3, padding=1, bias=False))
        self.body = nn.Sequential(*firstblock)

        self.CA = CALayer(n_feat, reduction, bias=bias)

    def forward(self, x):

        res = self.body(x)
        res = self.CA(res)
        res += x
        return res


##########################################################################
## Supervised Attention Module
class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size, bias):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, 3, kernel_size, bias=bias)
        self.conv3 = conv(3, n_feat, kernel_size, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1 * x2
        x1 = x1 + x

        return x1, img


##########################################################################
## U-Net

class Encoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff):
        super(Encoder, self).__init__()

        self.encoder_level1 = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.encoder_level2 = [CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act) for _ in
                               range(2)]
        self.encoder_level3 = [CAB(n_feat + (scale_unetfeats * 2), kernel_size, reduction, bias=bias, act=act) for _ in
                               range(2)]

        self.encoder_level1 = nn.Sequential(*self.encoder_level1)
        self.encoder_level2 = nn.Sequential(*self.encoder_level2)
        self.encoder_level3 = nn.Sequential(*self.encoder_level3)

        self.down12 = DownSample(n_feat, scale_unetfeats)
        self.down23 = DownSample(n_feat + scale_unetfeats, scale_unetfeats)

        # Cross Stage Feature Fusion (CSFF)
        if csff:
            self.csff_enc1 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
            self.csff_enc2 = nn.Conv2d(n_feat + scale_unetfeats, n_feat + scale_unetfeats, kernel_size=1, bias=bias)
            self.csff_enc3 = nn.Conv2d(n_feat + (scale_unetfeats * 2), n_feat + (scale_unetfeats * 2), kernel_size=1,
                                       bias=bias)

            self.csff_dec1 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
            self.csff_dec2 = nn.Conv2d(n_feat + scale_unetfeats, n_feat + scale_unetfeats, kernel_size=1, bias=bias)
            self.csff_dec3 = nn.Conv2d(n_feat + (scale_unetfeats * 2), n_feat + (scale_unetfeats * 2), kernel_size=1,
                                       bias=bias)

    def forward(self, x, encoder_outs=None, decoder_outs=None):
        enc1 = self.encoder_level1(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc1 = enc1 + self.csff_enc1(encoder_outs[0]) + self.csff_dec1(decoder_outs[0])

        x = self.down12(enc1)

        enc2 = self.encoder_level2(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc2 = enc2 + self.csff_enc2(encoder_outs[1]) + self.csff_dec2(decoder_outs[1])

        x = self.down23(enc2)

        enc3 = self.encoder_level3(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc3 = enc3 + self.csff_enc3(encoder_outs[2]) + self.csff_dec3(decoder_outs[2])

        return [enc1, enc2, enc3]


class Decoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats):
        super(Decoder, self).__init__()

        self.decoder_level1 = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.decoder_level2 = [CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act) for _ in
                               range(2)]
        self.decoder_level3 = [CAB(n_feat + (scale_unetfeats * 2), kernel_size, reduction, bias=bias, act=act) for _ in
                               range(2)]

        self.decoder_level1 = nn.Sequential(*self.decoder_level1)
        self.decoder_level2 = nn.Sequential(*self.decoder_level2)
        self.decoder_level3 = nn.Sequential(*self.decoder_level3)

        self.skip_attn1 = CAB(n_feat, kernel_size, reduction, bias=bias, act=act)
        self.skip_attn2 = CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act)

        self.up21 = SkipUpSample(n_feat, scale_unetfeats)
        self.up32 = SkipUpSample(n_feat + scale_unetfeats, scale_unetfeats)

    def forward(self, outs):
        enc1, enc2, enc3 = outs
        dec3 = self.decoder_level3(enc3)

        x = self.up32(dec3, self.skip_attn2(enc2))
        dec2 = self.decoder_level2(x)

        x = self.up21(dec2, self.skip_attn1(enc1))
        dec1 = self.decoder_level1(x)

        return [dec1, dec2, dec3]


##########################################################################
##---------- Resizing Modules ----------
class DownSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(DownSample, self).__init__()
        # self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
        #                           nn.Conv2d(in_channels, in_channels + s_factor, 1, stride=1, padding=0, bias=False))
        self.down = nn.Sequential(nn.MaxPool2d(2),
                                  nn.Conv2d(in_channels, in_channels + s_factor, 1, stride=1, padding=0, bias=False))
    def forward(self, x):
        x = self.down(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels + s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x


class SkipUpSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(SkipUpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels + s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x, y):
        x = self.up(x)
        x = x + y
        return x


##########################################################################
## Original Resolution Block (ORB)
class ORB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, num_cab):
        super(ORB, self).__init__()
        modules_body = []
        modules_body = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(num_cab)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)

class CA_SA(nn.Module): #
    def __init__(self, features, kernel_size, reduction):
        super(CA_SA, self).__init__()

        self.prelu = nn.PReLU()

        body = []
        body.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, bias=False))
        body.append(nn.PReLU())
        body.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, bias=False))
        self.body = nn.Sequential(*body)

        sa_attention = []
        sa_attention.append(nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, padding=0, bias=False))
        sa_attention.append(nn.PReLU())
        sa_attention.append(nn.Sigmoid())
        self.sab = nn.Sequential(*sa_attention)

        self.compress = ChannelPool()

        ch_attention = []
        ch_attention.append(nn.AdaptiveAvgPool2d(1))
        ch_attention.append(nn.Conv2d(in_channels=features, out_channels=features // reduction, kernel_size=1, padding=0, bias=False))
        ch_attention.append(nn.PReLU())
        ch_attention.append(nn.Conv2d(in_channels=features // reduction, out_channels=features, kernel_size=1, padding=0, bias=False))
        ch_attention.append(nn.Sigmoid())
        self.cab = nn.Sequential(*ch_attention)

        self.conv1x1 = nn.Conv2d(in_channels=features*2, out_channels=features, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        data = self.body(x)

        # Channel Attention
        ca_data = self.cab(data) * data
        # Spatial Attention
        sa_data = self.sab(self.compress(ca_data)) * ca_data

        # Global Residual
        out = sa_data + x

        return out


class DAB(nn.Module): # Deformed Convolution Attention Block
    def __init__(self, features, kernel_size, reduction, bias, act):
        super(DAB, self).__init__()
        groups = 8
        kernel_size = 3

        self.prelu = nn.PReLU()
        self.offset_conv1 = nn.Conv2d(features, 2*kernel_size*kernel_size, kernel_size=3, stride=1, padding=1, bias=True)
        self.deconv1 = dc.DeformConv2d(features, features, kernel_size=3, stride=1, padding=1, dilation=1,
                                   groups=groups)
        self.conv = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.softmax = nn.Softmax(-1)

    def forward(self, x):
        residual = x
        # deform conv
        offset1 = self.prelu(self.offset_conv1(x))
        feat_deconv1 = self.deconv1(x, offset1)

        # attention
        atten_conv = self.conv(x)
        atten_feat = self.softmax(atten_conv)

        out = atten_feat * feat_deconv1
        out = out + residual

        return out

class DCCA(nn.Module): # Deformed Convolution Attention Block
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats):
        super(DCCA, self).__init__()
        self.dab1 = DAB(n_feat, kernel_size, reduction, bias=bias, act=act)
        self.dab2 = DAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act)
        self.dab3 = DAB(n_feat + (scale_unetfeats * 2), kernel_size, reduction, bias=bias, act=act)

        self.cab1 = CAB(n_feat, kernel_size, reduction, bias=bias, act=act)
        self.cab2 = CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act)
        self.cab3 = CAB(n_feat + (scale_unetfeats * 2), kernel_size, reduction, bias=bias, act=act)

    def forward(self, x):
        enc1, enc2, enc3 = x

        res3 = self.dab3(enc3)
        res3 = self.cab3(res3) + enc3
        res2 = self.dab2(enc2)
        res2 = self.cab2(res2) + enc2
        res1 = self.dab1(enc1)
        res1 = self.cab1(res1) + enc1

        return [res1, res2, res3]

##########################################################################
class ORSNet(nn.Module):
    def __init__(self, n_feat, scale_orsnetfeats, kernel_size, reduction, act, bias, scale_unetfeats, num_cab):
        super(ORSNet, self).__init__()

        self.orb1 = ORB(n_feat + scale_orsnetfeats, kernel_size, reduction, act, bias, num_cab)
        self.orb2 = ORB(n_feat + scale_orsnetfeats, kernel_size, reduction, act, bias, num_cab)
        self.orb3 = ORB(n_feat + scale_orsnetfeats, kernel_size, reduction, act, bias, num_cab)

        self.up_enc1 = UpSample(n_feat, scale_unetfeats)
        self.up_dec1 = UpSample(n_feat, scale_unetfeats)

        self.up_enc2 = nn.Sequential(UpSample(n_feat + scale_unetfeats, scale_unetfeats),
                                     UpSample(n_feat, scale_unetfeats))
        self.up_dec2 = nn.Sequential(UpSample(n_feat + scale_unetfeats, scale_unetfeats),
                                     UpSample(n_feat, scale_unetfeats))

        self.conv_enc1 = nn.Conv2d(n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_enc2 = nn.Conv2d(n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_enc3 = nn.Conv2d(n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)

        self.conv_dec1 = nn.Conv2d(n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_dec2 = nn.Conv2d(n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_dec3 = nn.Conv2d(n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)

    def forward(self, x, encoder_outs, decoder_outs):
        x = self.worb1(x)
        x = x + self.conv_enc1(encoder_outs[0]) + self.conv_dec1(decoder_outs[0])

        x = self.worb2(x)
        x = x + self.conv_enc2(self.up_enc1(encoder_outs[1])) + self.conv_dec2(self.up_dec1(decoder_outs[1]))

        x = self.worb3(x)
        x = x + self.conv_enc3(self.up_enc2(encoder_outs[2])) + self.conv_dec3(self.up_dec2(decoder_outs[2]))

        return x

##########################################################################
class EdgeMoudule(nn.Module):
    def __init__(self, channels, features):
        super(EdgeMoudule, self).__init__()

        edgeblock = []
        edgeblock.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=3, padding=1, bias=False))
        edgeblock.append(nn.PReLU())
        for _ in range(4):
            edgeblock.append(
                nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False))
            edgeblock.append(nn.PReLU())
        edgeblock.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=3, padding=1, bias=False))

        self.edgeblock = nn.Sequential(*edgeblock)


    def forward(self, x):
        edge = self.edgeblock(x)

        return edge

##########################################################################
class SKSAK(nn.Module):
    def __init__(self, channels=3, features=96, scale_unetfeats=32, scale_orsnetfeats=32, num_cab=8, kernel_size=3,
                 reduction=4, bias=False):
        super(SKSAK, self).__init__()

        self.edge_module = EdgeMoudule(channels, features)

        act = nn.PReLU()

        self.feature_extract = common.sksak_edgecomb(channels, features)

        # Cross Stage Feature Fusion (CSFF)
        self.stage1_encoder = Encoder(features, kernel_size, reduction, act, bias, scale_unetfeats, csff=False)
        self.stage1_decoder = Decoder(features, kernel_size, reduction, act, bias, scale_unetfeats)

        self.stage2_encoder = Encoder(features, kernel_size, reduction, act, bias, scale_unetfeats, csff=True)
        self.stage2_decoder = Decoder(features, kernel_size, reduction, act, bias, scale_unetfeats)

        # self.stage3_orsnet = CBAMNet(features, scale_orsnetfeats, kernel_size, reduction, act, bias, scale_unetfeats, num_cab)
        self.stage3_encoder = Encoder(features, kernel_size, reduction, act, bias, scale_unetfeats, csff=True)
        self.stage3_decoder = Decoder(features, kernel_size, reduction, act, bias, scale_unetfeats)

        self.unet_plat = DCCA(features, kernel_size, reduction, bias, act, scale_unetfeats)

        self.sam12 = SAM(features, kernel_size=1, bias=bias)
        self.sam23 = SAM(features, kernel_size=1, bias=bias)

        self.upscale = nn.Sequential(
            nn.Conv2d(features, features * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2)
        )

        self.stage_upscale = nn.Sequential(
            nn.Conv2d(channels, features * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2)
        )

        self.ch2feat = conv(channels, features, kernel_size, bias=bias)
        self.concat12 = conv(features * 2, features, kernel_size, bias=bias)
        self.concat23 = conv(features, features + scale_orsnetfeats, kernel_size, bias=bias)
        self.tail = conv(features, channels, kernel_size, bias=bias)

    def forward(self, x3_img):
        # Original-resolution Image for Stage 3
        H = x3_img.size(2)
        W = x3_img.size(3)

        # Edge Module
        edge = x3_img - self.edge_module(x3_img)

        up_edge = self.stage_upscale(edge)
        edge_img = self.tail(up_edge)

        # Multi-Patch Hierarchy: Split Image into four non-overlapping patches

        # Two Patches for Stage 2
        x2top_img = x3_img[:, :, 0:int(H / 2), :]
        x2bot_img = x3_img[:, :, int(H / 2):H, :]
        x2top_edge = edge[:, :, 0:int(H / 2), :]
        x2bot_edge = edge[:, :, int(H / 2):H, :]

        # Four Patches for Stage 1
        x1ltop_img = x2top_img[:, :, :, 0:int(W / 2)]
        x1rtop_img = x2top_img[:, :, :, int(W / 2):W]
        x1lbot_img = x2bot_img[:, :, :, 0:int(W / 2)]
        x1rbot_img = x2bot_img[:, :, :, int(W / 2):W]

        x1ltop_edge = x2top_edge[:, :, :, 0:int(W / 2)]
        x1rtop_edge = x2top_edge[:, :, :, int(W / 2):W]
        x1lbot_edge = x2bot_edge[:, :, :, 0:int(W / 2)]
        x1rbot_edge = x2bot_edge[:, :, :, int(W / 2):W]

        ##-------------------------------------------
        ##-------------- Stage 1---------------------
        ##-------------------------------------------
        ## Compute Shallow Features
        x1ltop = self.feature_extract(x1ltop_img, x1ltop_edge)
        x1rtop = self.feature_extract(x1rtop_img, x1rtop_edge)
        x1lbot = self.feature_extract(x1lbot_img, x1lbot_edge)
        x1rbot = self.feature_extract(x1rbot_img, x1rbot_edge)

        ## Process features of all 4 patches with Encoder of Stage 1
        feat1_ltop = self.stage1_encoder(x1ltop)
        feat1_rtop = self.stage1_encoder(x1rtop)
        feat1_lbot = self.stage1_encoder(x1lbot)
        feat1_rbot = self.stage1_encoder(x1rbot)

        ## Concat deep features
        feat1_top = [torch.cat((k, v), 3) for k, v in zip(feat1_ltop, feat1_rtop)]
        feat1_bot = [torch.cat((k, v), 3) for k, v in zip(feat1_lbot, feat1_rbot)]

        stage1_plat1 = self.unet_plat(feat1_top)
        stage1_plat2 = self.unet_plat(feat1_bot)

        ## Pass features through Decoder of Stage 1
        res1_top = self.stage1_decoder(stage1_plat1)
        res1_bot = self.stage1_decoder(stage1_plat2)

        ## Apply Supervised Attention Module (SAM)
        x2top_samfeats, stage1_img_top = self.sam12(res1_top[0], x2top_img)
        x2bot_samfeats, stage1_img_bot = self.sam12(res1_bot[0], x2bot_img)

        ## Output image at Stage 1
        stage1 = torch.cat([stage1_img_top, stage1_img_bot], 2)

        up_stage1 = self.stage_upscale(stage1)
        stage1_img = self.tail(up_stage1)

        ##-------------------------------------------
        ##-------------- Stage 2---------------------
        ##-------------------------------------------
        ## Compute Shallow Features
        x2top = self.feature_extract(x2top_img, x2top_edge)
        x2bot = self.feature_extract(x2bot_img, x2bot_edge)

        ## Concatenate SAM features of Stage 1 with shallow features of Stage 2
        # x2top_cat = self.concat12(torch.cat([x2top, x2top_samfeats], 1))
        # x2bot_cat = self.concat12(torch.cat([x2bot, x2bot_samfeats], 1))

        x2top_cat = x2top + x2top_samfeats
        x2bot_cat = x2bot + x2bot_samfeats

        ## Process features of both patches with Encoder of Stage 2
        feat2_top = self.stage2_encoder(x2top_cat, feat1_top, res1_top)
        feat2_bot = self.stage2_encoder(x2bot_cat, feat1_bot, res1_bot)

        ## Concat deep features
        feat2 = [torch.cat((k, v), 2) for k, v in zip(feat2_top, feat2_bot)]

        stage2_plat = self.unet_plat(feat2)

        ## Pass features through Decoder of Stage 2
        res2 = self.stage2_decoder(stage2_plat)

        ## Apply SAM
        x3_samfeats, stage2 = self.sam23(res2[0], x3_img)

        up_stage2 = self.stage_upscale(stage2)
        stage2_img = self.tail(up_stage2)

        ##-------------------------------------------
        ##-------------- Stage 3---------------------
        ##-------------------------------------------
        ## Compute Shallow Features
        x3_img_feat = self.feature_extract(x3_img, edge)
        # x3 = self.shallow_feat3(x3_img_edge)

        ## Concatenate SAM features of Stage 2 with shallow features of Stage 3
        # x3_cat = self.concat23(torch.cat([x3, x3_samfeats], 1))
        x3_cat = x3_img_feat + x3_samfeats
        feat3_enc = self.stage3_encoder(x3_cat, feat2, res2)
        stage3_plat = self.unet_plat(feat3_enc)
        feat3_dec = self.stage3_decoder(stage3_plat)

        # x3_cat = self.stage3_orsnet(x3_cat, feat2, res2)

        up_out = self.upscale(feat3_dec[0])
        out = self.tail(up_out)

        return [out, stage2_img, stage1_img, edge_img]
