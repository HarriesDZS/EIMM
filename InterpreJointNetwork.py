

import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from thop import profile


class ConvBN(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv', torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn', torch.nn.BatchNorm2d(out_planes))
            torch.nn.init.constant_(self.bn.weight, 1)
            torch.nn.init.constant_(self.bn.bias, 0)


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0.):
        super().__init__()
        self.dwconv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        x = input + self.drop_path(x)
        return x

class Encoder(nn.Module):
    def __init__(self, input_channel, init_channel):
        super().__init__()
        self.first_conv = ConvBN(in_planes=input_channel, out_planes=init_channel)

        self.star_conv1 = Block(dim=init_channel)
        self.down_conv1 = ConvBN(init_channel, 2 * init_channel, 3, 2, 1)
        self.star_conv2 = Block(dim=2 * init_channel)
        self.down_conv2 = ConvBN(2 * init_channel, 4 * init_channel, 3, 2, 1)
        self.star_conv3 = Block(dim=4 * init_channel)
        self.down_conv3 = ConvBN(4 * init_channel, 8 * init_channel, 3, 2, 1)
        self.star_conv4 = Block(dim=8 * init_channel)
        self.dconv_down4 = ConvBN(8 * init_channel, 8
                                  * init_channel, 3, 2, 1)
        self.star_conv5 = Block(dim=8 * init_channel)

    def forward(self, x):
        conv1 = self.star_conv1(self.first_conv(x))
        x = self.down_conv1(conv1)

        conv2 = self.star_conv2(x)
        x = self.down_conv2(conv2)

        conv3 = self.star_conv3(x)
        x = self.down_conv3(conv3)

        conv4 = self.star_conv4(x)
        x = self.dconv_down4(conv4)

        shared_feature = self.star_conv5(x)
        return conv1,conv2,conv3,conv4,shared_feature


class InterpreStarJoinNet(nn.Module):

    def __init__(self, input_channel, seg_class, cls_class, init_channel = 32):
        super().__init__()

        #1、共享特征提取分支
        self.encoder = Encoder(input_channel=input_channel, init_channel=init_channel)


        #分割分支
        self.up_sample4 = nn.ConvTranspose2d(8*init_channel, 8*init_channel, stride=2, kernel_size=2)
        self.up_sample3 = nn.ConvTranspose2d(8*init_channel, 8*init_channel, stride=2, kernel_size=2)
        self.up_sample2 = nn.ConvTranspose2d(4*init_channel, 4*init_channel, stride=2, kernel_size=2)
        self.up_sample1 = nn.ConvTranspose2d(2*init_channel, 2*init_channel, stride=2, kernel_size=2)

        self.up_conv1 = ConvBN(16*init_channel, 8*init_channel)
        self.star_seg1 = Block(dim=8*init_channel)
        self.up_conv2 = ConvBN(12*init_channel, 4*init_channel)
        self.star_seg2 = Block(dim=4*init_channel)
        self.up_conv3 = ConvBN(6*init_channel, 2*init_channel)
        self.star_seg3 = Block(dim=2*init_channel)
        self.up_conv4 = ConvBN(3*init_channel, init_channel)
        self.star_seg4 = Block(dim=init_channel)

        self.seg_head = ConvBN(init_channel, seg_class)

        #边缘预测分支
        self.seg_edge = ConvBN(init_channel, seg_class)

        #肿瘤重建分支
        self.recover_branch = nn.Sequential(
            nn.ConvTranspose2d(8 * init_channel, 8 * init_channel, stride=2, kernel_size=2),
            Block(dim=8 * init_channel),
            nn.ConvTranspose2d(8 * init_channel, 4 * init_channel, stride=2, kernel_size=2),
            Block(dim=4 * init_channel),
            nn.ConvTranspose2d(4 * init_channel, 2 * init_channel, stride=2, kernel_size=2),
            Block(dim=2 * init_channel),
            nn.ConvTranspose2d(2 * init_channel, init_channel, stride=2, kernel_size=2),
            Block(dim = init_channel),
            ConvBN(init_channel, input_channel)
        )

        #基于MLP对特征进行映射，并且基于映射之后的特征进行对比学习,特征为128维
        self.MLP = nn.Sequential(
            nn.Linear(8*init_channel*32*32, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 128)
        )


        #分类分支
        self.cls_conv1 = ConvBN(8*init_channel, 4*init_channel)
        self.star_cls1 = Block(dim=4*init_channel)
        self.cls_conv2 = ConvBN(4*init_channel, 2*init_channel)
        self.star_cls2 = Block(dim=2*init_channel)
        self.norm = nn.BatchNorm2d(2*init_channel)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.cls_head = nn.Linear(2*init_channel, cls_class)


        self.edge_factor = nn.Linear(1, 1)
        self.recover_factor = nn.Linear(1, 1)


    def forward(self, x, x_aug, x_same_patient, x_other_class):

        conv1, conv2, conv3, conv4, shared_feature = self.encoder(x)

        #seg
        x = self.up_sample4(shared_feature)
        x = torch.cat([x, conv4], dim=1)
        x = self.up_conv1(x)
        x = self.star_seg1(x)

        x = self.up_sample3(x)
        x = torch.cat([x, conv3], dim=1)
        x = self.up_conv2(x)
        x = self.star_seg2(x)

        x = self.up_sample2(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.up_conv3(x)
        x = self.star_seg3(x)

        x = self.up_sample1(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.up_conv4(x)
        x = self.star_seg4(x)

        seg = self.seg_head(x)
        seg_edge = self.seg_edge(x)
        seg_binary = seg.argmax(dim=1)
        seg_edge_binary = seg_edge.argmax(dim=1)

        edge_value = nn.functional.softmax(seg_edge, dim=1)
        edge_value = -torch.sum(edge_value[:,1,:] * torch.log2(edge_value[:,1,:] + 1e-8), dim=[1,2])
        edge_value = edge_value / seg_edge_binary.sum(axis=[1,2])
        edge_value = nn.functional.sigmoid(edge_value)



        #重建分支
        seg_binary = seg_binary.unsqueeze(1)
        seg_binary = seg_binary.type(torch.FloatTensor).cuda(device=0)
        small_seg_binary = nn.functional.interpolate(seg_binary, size=(shared_feature.shape[2], shared_feature.shape[3]),
                                                     mode="bilinear", align_corners=True)
        recover = self.recover_branch(shared_feature * small_seg_binary)
        recover_value = torch.sum(torch.abs((recover * seg_binary) - (x * seg_binary)), dim=[1, 2, 3]) / (seg_binary.sum(axis=[1, 2, 3]) + 1e-8)


        # cls
        x = self.star_cls1(self.cls_conv1(shared_feature))
        x = self.star_cls2(self.cls_conv2(x))
        cls = self.cls_head(self.avgpool(self.norm(x)).squeeze(2).squeeze(2))
        factor = torch.concat([self.edge_factor(recover_value.view(recover_value.shape[0], 1)),
                               self.edge_factor(edge_value.view(edge_value.shape[0], 1))], dim=1)
        cls = cls + factor



        #对比的几个特征
        _, _, _, _, x_aug_feature = self.encoder(x_aug)
        _, _, _, _, x_same_patient_feature = self.encoder(x_same_patient)
        _, _, _, _, x_othert_class_feature = self.encoder(x_other_class)
        x_feature = self.MLP(shared_feature.view(shared_feature.size(0), -1))
        x_aug_feature = self.MLP(x_aug_feature.view(x_aug_feature.size(0), -1))
        x_same_patient_feature = self.MLP(x_same_patient_feature.view(x_same_patient_feature.size(0), -1))
        x_othert_class_feature = self.MLP(x_othert_class_feature.view(x_othert_class_feature.size(0), -1))

        return seg, cls, recover, seg_edge, x_feature, x_aug_feature, x_same_patient_feature, x_othert_class_feature

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0, 0.05)
                if m.bias is not None:
                    m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


