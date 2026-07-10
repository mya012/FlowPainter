import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torchvision import ops
import math

# ============ 辅助函数 ============
def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, activation=True):
    if activation:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            nn.LeakyReLU(0.1))
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True))


def predict_flow(in_planes):
    return nn.Conv2d(in_planes, 2, kernel_size=3, stride=1, padding=1, bias=True)


def predict_mask(in_planes):
    return nn.Conv2d(in_planes, 1, kernel_size=3, stride=1, padding=1, bias=True)


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=True)


def deformable_conv(in_planes, out_planes, kernel_size=3, strides=1, padding=1, use_bias=True):
    return ops.DeformConv2d(in_planes, out_planes, kernel_size, strides, padding, bias=use_bias)


def upsample_kernel2d(w, device):
    c = w // 2
    kernel = 1 - torch.abs(c - torch.arange(w, dtype=torch.float32, device=device)) / (c + 1)
    kernel = kernel.repeat(w).view(w,-1) * kernel.unsqueeze(1)
    return kernel.view(1, 1, w, w)


def downsample_kernel2d(w, device):
    kernel = ((w + 1) - torch.abs(w - torch.arange(w * 2 + 1, dtype=torch.float32, device=device))) / (2 * w + 1)
    kernel = kernel.repeat(w).view(w,-1) * kernel.unsqueeze(1)
    return kernel.view(1, 1, w * 2 + 1, w * 2 + 1)


def Upsample(img, factor):
    if factor == 1:
        return img
    B, C, H, W = img.shape
    batch_img = img.view(B*C, 1, H, W)
    batch_img = F.pad(batch_img, [0, 1, 0, 1], mode='replicate')
    kernel = upsample_kernel2d(factor * 2 - 1, img.device)
    upsamp_img = F.conv_transpose2d(batch_img, kernel, stride=factor, padding=(factor-1))
    upsamp_img = upsamp_img[:, :, : -1, :-1]
    _, _, H_up, W_up = upsamp_img.shape
    return upsamp_img.view(B, C, H_up, W_up)


def Downsample(img, factor):
    if factor == 1:
        return img
    B, C, H, W = img.shape
    batch_img = img.view(B*C, 1, H, W)
    kernel = downsample_kernel2d(factor // 2, img.device)
    upsamp_img = F.conv2d(batch_img, kernel, stride=factor, padding=factor//2)
    upsamp_nom = F.conv2d(torch.ones_like(batch_img), kernel, stride=factor, padding=factor//2)
    _, _, H_up, W_up = upsamp_img.shape
    upsamp_img = upsamp_img.view(B, C, H_up, W_up)
    upsamp_nom = upsamp_nom.view(B, C, H_up, W_up)
    return upsamp_img / upsamp_nom


def backward_warp(img, flow):
    """用于infer方法的warp函数"""
    B, _, H, W = flow.shape
    xx = torch.linspace(-1.0, 1.0, W).view(1, 1, 1, W).expand(B, -1, H, -1)
    yy = torch.linspace(-1.0, 1.0, H).view(1, 1, H, 1).expand(B, -1, -1, W)
    grid = torch.cat([xx, yy], 1).to(img)
    flow_ = torch.cat([flow[:, 0:1, :, :] / ((W - 1.0) / 2.0), 
                       flow[:, 1:2, :, :] / ((H - 1.0) / 2.0)], 1)
    grid_ = (grid + flow_).permute(0, 2, 3, 1)
    output = F.grid_sample(input=img, grid=grid_, mode='bilinear', 
                          padding_mode='border', align_corners=True)
    return output


def pad_to_multiple(x, multiple=64):
    """将输入填充到multiple的倍数，MaskFlownet需要64的倍数"""
    h, w = x.shape[2], x.shape[3]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')
    
    return x, (h, w, pad_h, pad_w)


def unpad(x, original_size):
    """移除填充"""
    h, w, pad_h, pad_w = original_size
    if pad_h > 0 or pad_w > 0:
        x = x[:, :, :h, :w]
    return x


# ============ MaskFlownet_S兼容版本 ============
class MaskFlownet_S_Compat(nn.Module):
    """
    修改后的MaskFlownet_S，使其具有与目标代码相同的接口格式
    包含内置的Correlation类，避免import问题
    """
    def __init__(self, config=None, **kwargs):
        super(MaskFlownet_S_Compat, self).__init__()
        self.scale = 20. * 1.0
        md = 4
        self.md = md
        self.strides = [64, 32, 16, 8, 4]
        self.deform_bias = True
        self.upfeat_ch = [16, 16, 16, 16]

        # 特征提取网络
        self.conv1a  = conv(3,   16, kernel_size=3, stride=2)
        self.conv1b = conv(16, 16, kernel_size=3, stride=1)
        self.conv1c  = conv(16, 16, kernel_size=3, stride=1)
        self.conv2a  = conv(16,  32, kernel_size=3, stride=2)
        self.conv2b = conv(32, 32, kernel_size=3, stride=1)
        self.conv2c  = conv(32, 32, kernel_size=3, stride=1)
        self.conv3a  = conv(32,  64, kernel_size=3, stride=2)
        self.conv3b = conv(64, 64, kernel_size=3, stride=1)
        self.conv3c  = conv(64, 64, kernel_size=3, stride=1)
        self.conv4a  = conv(64,  96, kernel_size=3, stride=2)
        self.conv4b = conv(96, 96, kernel_size=3, stride=1)
        self.conv4c  = conv(96, 96, kernel_size=3, stride=1)
        self.conv5a  = conv(96, 128, kernel_size=3, stride=2)
        self.conv5b = conv(128, 128, kernel_size=3, stride=1)
        self.conv5c  = conv(128, 128, kernel_size=3, stride=1)
        self.conv6a = conv(128, 196, kernel_size=3, stride=2)
        self.conv6b  = conv(196, 196, kernel_size=3, stride=1)
        self.conv6c  = conv(196, 196, kernel_size=3, stride=1)

        # 使用内置的Correlation类
        # self.corr = Correlation(pad_size=md, kernel_size=1, max_displacement=md, 
                                # stride1=1, stride2=1, corr_multiply=1)
        self.corr = self.corr
        self.leakyRELU = nn.LeakyReLU(0.1)

        nd = (2*md+1)**2
        dd = np.cumsum([128,128,96,64,32])

        # 解码器网络
        od = nd
        self.conv6_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv6_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.conv6_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.conv6_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.conv6_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)
        self.pred_flow6 = predict_flow(od + dd[4])
        self.pred_mask6 = predict_mask(od + dd[4])
        self.upfeat5 = deconv(od+dd[4], self.upfeat_ch[0], kernel_size=4, stride=2, padding=1)

        od = nd+128+18
        self.conv5_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv5_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.conv5_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.conv5_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.conv5_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)
        self.pred_flow5 = predict_flow(od + dd[4])
        self.pred_mask5 = predict_mask(od + dd[4])
        self.upfeat4 = deconv(od+dd[4], self.upfeat_ch[1], kernel_size=4, stride=2, padding=1)

        od = nd+96+18
        self.conv4_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv4_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.conv4_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.conv4_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.conv4_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)
        self.pred_flow4 = predict_flow(od + dd[4])
        self.pred_mask4 = predict_mask(od + dd[4])
        self.upfeat3 = deconv(od+dd[4], self.upfeat_ch[2], kernel_size=4, stride=2, padding=1)

        od = nd+64+18
        self.conv3_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv3_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.conv3_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.conv3_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.conv3_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)
        self.pred_flow3 = predict_flow(od + dd[4])
        self.pred_mask3 = predict_mask(od + dd[4])
        self.upfeat2 = deconv(od+dd[4], self.upfeat_ch[3], kernel_size=4, stride=2, padding=1)

        od = nd+32+18
        self.conv2_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv2_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.conv2_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.conv2_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.conv2_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)
        self.pred_flow2 = predict_flow(od + dd[4])

        # 上下文网络
        self.dc_conv1 = conv(od+dd[4], 128, kernel_size=3, stride=1, padding=1,  dilation=1)
        self.dc_conv2 = conv(128,      128, kernel_size=3, stride=1, padding=2,  dilation=2)
        self.dc_conv3 = conv(128,      128, kernel_size=3, stride=1, padding=4,  dilation=4)
        self.dc_conv4 = conv(128,      96,  kernel_size=3, stride=1, padding=8,  dilation=8)
        self.dc_conv5 = conv(96,       64,  kernel_size=3, stride=1, padding=16, dilation=16)
        self.dc_conv6 = conv(64,       32,  kernel_size=3, stride=1, padding=1,  dilation=1)
        self.dc_conv7 = predict_flow(32)

        # 可变形卷积
        self.deform5 = deformable_conv(128, 128)
        self.deform4 = deformable_conv(96, 96)
        self.deform3 = deformable_conv(64, 64)
        self.deform2 = deformable_conv(32, 32)

        # 特征融合
        self.conv5f = conv(16, 128, kernel_size=3, stride=1, padding=1, activation=False)
        self.conv4f = conv(16, 96, kernel_size=3, stride=1, padding=1, activation=False)
        self.conv3f = conv(16, 64, kernel_size=3, stride=1, padding=1, activation=False)
        self.conv2f = conv(16, 32, kernel_size=3, stride=1, padding=1, activation=False)

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

    def corr(self, refimg_fea, targetimg_fea):
        maxdisp=4
        b,c,h,w = refimg_fea.shape
        # 通过F.unfold取出f2的窗口范围d=2*maxdisp+1内的特征向量，
        # 这里由于F.unfold的步长默认为1，所以每个WH都能提取出周围d*d的特征向量，
        # 因此可以view到（b,c,2*maxdisp+1, 2*maxdisp+1,h,w）
        # ps：个人认为第二个 2*maxdisp+1**2 的 次方是 个错误，应也是2*maxdisp+1，但不影响
        targetimg_fea = F.unfold(targetimg_fea, (2*maxdisp+1,2*maxdisp+1), padding=maxdisp).view(b,c,2*maxdisp+1, 2*maxdisp+1,h,w)
        # 对f1扩维， ps： targetimg_fea后的view应该无效
        cost = refimg_fea.view(b,c,h,w)[:,:,np.newaxis, np.newaxis]*targetimg_fea.view(b,c,2*maxdisp+1, 2*maxdisp+1,h,w)
        cost = cost.sum(1)

        b, ph, pw, h, w = cost.size()
        cost = cost.view(b, ph * pw, h, w)/refimg_fea.size(1)
        return cost


    def forward(self, img0, img1):
        """
        修改后的forward方法，与目标代码格式一致
        输入: img0, img1 [B, 3, H, W]
        输出: flow, mask [B, 2, H, W], [B, 1, H, W]
        """
        # # 保存原始尺寸并填充到64的倍数（MaskFlownet需要）
        # original_h, original_w = img0.shape[2], img0.shape[3]
        # img0_padded, pad_info0 = pad_to_multiple(img0, 64)
        # img1_padded, pad_info1 = pad_to_multiple(img1, 64)
        
        # im1 = img0_padded
        # im2 = img1_padded

        # 插值到64的倍数
        original_h, original_w = img0.shape[2], img0.shape[3]
        inter_h, inter_w = math.ceil(original_h / 64) * 64, math.ceil(original_w / 64) * 64
        if inter_h != original_h or inter_w != original_w:
            im1 = F.interpolate(img0, size=(inter_h, inter_w), mode='bilinear', align_corners=True)
            im2 = F.interpolate(img1, size=(inter_h, inter_w), mode='bilinear', align_corners=True)
        else:
            im1 = img0
            im2 = img1

        # 特征提取
        c11 = self.conv1c(self.conv1b(self.conv1a(im1)))
        c21 = self.conv1c(self.conv1b(self.conv1a(im2)))
        c12 = self.conv2c(self.conv2b(self.conv2a(c11)))
        c22 = self.conv2c(self.conv2b(self.conv2a(c21)))
        c13 = self.conv3c(self.conv3b(self.conv3a(c12)))
        c23 = self.conv3c(self.conv3b(self.conv3a(c22)))
        c14 = self.conv4c(self.conv4b(self.conv4a(c13)))
        c24 = self.conv4c(self.conv4b(self.conv4a(c23)))
        c15 = self.conv5c(self.conv5b(self.conv5a(c14)))
        c25 = self.conv5c(self.conv5b(self.conv5a(c24)))
        c16 = self.conv6c(self.conv6b(self.conv6a(c15)))
        c26 = self.conv6c(self.conv6b(self.conv6a(c25)))
        # import ipdb; ipdb.set_trace()

        # 相关性计算（使用内置Correlation类）
        corr6 = self.corr(c16, c26)
        corr6 = self.leakyRELU(corr6)

        # 金字塔处理流程
        x = torch.cat((self.conv6_0(corr6), corr6),1)
        x = torch.cat((self.conv6_1(x), x),1)
        x = torch.cat((self.conv6_2(x), x),1)
        x = torch.cat((self.conv6_3(x), x),1)
        x = torch.cat((self.conv6_4(x), x),1)
        flow6 = self.pred_flow6(x)
        mask6 = self.pred_mask6(x)

        # 第5层
        feat5 = self.leakyRELU(self.upfeat5(x))
        flow5 = Upsample(flow6, 2)
        mask5 = Upsample(mask6, 2)
        warp5 = (flow5*self.scale/self.strides[1]).unsqueeze(1)
        warp5 = torch.repeat_interleave(warp5, 9, 1)
        S1, S2, S3, S4, S5 = warp5.shape
        warp5 = warp5.view(S1, S2*S3, S4, S5)
        warp5 = self.deform5(c25, warp5)
        tradeoff5 = feat5
        warp5 = (warp5 * torch.sigmoid(mask5)) + self.conv5f(tradeoff5)
        warp5 = self.leakyRELU(warp5)
        corr5 = self.corr(c15, warp5)
        corr5 = self.leakyRELU(corr5)
        x = torch.cat((corr5, c15, feat5, flow5), 1)
        x = torch.cat((self.conv5_0(x), x),1)
        x = torch.cat((self.conv5_1(x), x),1)
        x = torch.cat((self.conv5_2(x), x),1)
        x = torch.cat((self.conv5_3(x), x),1)
        x = torch.cat((self.conv5_4(x), x),1)
        flow5 = flow5 + self.pred_flow5(x)
        mask5 = self.pred_mask5(x)

        # 第4层
        feat4 = self.leakyRELU(self.upfeat4(x))
        flow4 = Upsample(flow5, 2)
        mask4 = Upsample(mask5, 2)
        warp4 = (flow4*self.scale/self.strides[2]).unsqueeze(1)
        warp4 = torch.repeat_interleave(warp4, 9, 1)
        S1, S2, S3, S4, S5 = warp4.shape
        warp4 = warp4.view(S1, S2*S3, S4, S5)
        warp4 = self.deform4(c24, warp4)
        tradeoff4 = feat4
        warp4 = (warp4 * torch.sigmoid(mask4)) + self.conv4f(tradeoff4)
        warp4 = self.leakyRELU(warp4)
        corr4 = self.corr(c14, warp4)
        corr4 = self.leakyRELU(corr4)
        x = torch.cat((corr4, c14, feat4, flow4), 1)
        x = torch.cat((self.conv4_0(x), x),1)
        x = torch.cat((self.conv4_1(x), x),1)
        x = torch.cat((self.conv4_2(x), x),1)
        x = torch.cat((self.conv4_3(x), x),1)
        x = torch.cat((self.conv4_4(x), x),1)
        flow4 = flow4 + self.pred_flow4(x)
        mask4 = self.pred_mask4(x)

        # 第3层
        feat3 = self.leakyRELU(self.upfeat3(x))
        flow3 = Upsample(flow4, 2)
        mask3 = Upsample(mask4, 2)
        warp3 = (flow3*self.scale/self.strides[3]).unsqueeze(1)
        warp3 = torch.repeat_interleave(warp3, 9, 1)
        S1, S2, S3, S4, S5 = warp3.shape
        warp3 = warp3.view(S1, S2*S3, S4, S5)
        warp3 = self.deform3(c23, warp3)
        tradeoff3 = feat3
        warp3 = (warp3 * torch.sigmoid(mask3)) + self.conv3f(tradeoff3)
        warp3 = self.leakyRELU(warp3)
        corr3 = self.corr(c13, warp3)
        corr3 = self.leakyRELU(corr3)
        x = torch.cat((corr3, c13, feat3, flow3), 1)
        x = torch.cat((self.conv3_0(x), x),1)
        x = torch.cat((self.conv3_1(x), x),1)
        x = torch.cat((self.conv3_2(x), x),1)
        x = torch.cat((self.conv3_3(x), x),1)
        x = torch.cat((self.conv3_4(x), x),1)
        flow3 = flow3 + self.pred_flow3(x)
        mask3 = self.pred_mask3(x)

        # 第2层
        feat2 = self.leakyRELU(self.upfeat2(x))
        flow2 = Upsample(flow3, 2)
        mask2 = Upsample(mask3, 2)
        warp2 = (flow2*self.scale/self.strides[4]).unsqueeze(1)
        warp2 = torch.repeat_interleave(warp2, 9, 1)
        S1, S2, S3, S4, S5 = warp2.shape
        warp2 = warp2.view(S1, S2*S3, S4, S5)
        warp2 = self.deform2(c22, warp2)
        tradeoff2 = feat2
        warp2 = (warp2 * torch.sigmoid(mask2)) + self.conv2f(tradeoff2)
        warp2 = self.leakyRELU(warp2)
        corr2 = self.corr(c12, warp2)
        corr2 = self.leakyRELU(corr2)
        x = torch.cat((corr2, c12, feat2, flow2), 1)
        x = torch.cat((self.conv2_0(x), x),1)
        x = torch.cat((self.conv2_1(x), x),1)
        x = torch.cat((self.conv2_2(x), x),1)
        x = torch.cat((self.conv2_3(x), x),1)
        x = torch.cat((self.conv2_4(x), x),1)
        flow2 = flow2 + self.pred_flow2(x)

        # 上下文网络细化
        x = self.dc_conv4(self.dc_conv3(self.dc_conv2(self.dc_conv1(x))))
        flow2 = flow2 + self.dc_conv7(self.dc_conv6(self.dc_conv5(x)))

        # import ipdb; ipdb.set_trace()

        # 使用最终的光流预测 (flow2)
        flow = flow2 * self.scale
        mask = torch.sigmoid(Upsample(mask2, 4))
        flow = F.interpolate(flow, scale_factor=4, mode='bilinear', align_corners=True)

        # import ipdb; ipdb.set_trace()

        if inter_h != original_h or inter_w != original_w:
            flow = F.interpolate(flow, size=(original_h, original_w), mode='bilinear', align_corners=True)
            mask = F.interpolate(mask, size=(original_h, original_w), mode='bilinear', align_corners=True)

            scale_factor_h = original_h / flow.shape[2]
            scale_factor_w = original_w / flow.shape[3]
            flow[:, 0] *= scale_factor_w
            flow[:, 1] *= scale_factor_h
        

        return flow, mask
        
        

        # mask = torch.sigmoid(Upsample(mask2, 4))

        # flow_2 = F.interpolate(flow2, scale_factor=4, mode='bilinear', align_corners=True)
        # flow_3 = F.interpolate(flow3, scale_factor=4, mode='bilinear', align_corners=True)
        # flow_4 = F.interpolate(flow4, scale_factor=4, mode='bilinear', align_corners=True)
        # flow_5 = F.interpolate(flow5, scale_factor=4, mode='bilinear', align_corners=True)
        # flow_6 = F.interpolate(flow6, scale_factor=4, mode='bilinear', align_corners=True)
        # predictions = [flow * self.scale for flow in [flow_2, flow_3, flow_4, flow_5, flow_6]]
        
        # return predictions, mask

    def infer(self, img0, img1, warp_img):
        """
        推理方法，与目标代码格式一致
        输入: img0, img1, warp_img
        输出: flow, mask, pred_img
        """
        flow, mask = self.forward(img0, img1)
        pred_img = backward_warp(warp_img, flow)
        return flow, mask, pred_img


# 保持原始类名不变
MaskFlownet_S = MaskFlownet_S_Compat
