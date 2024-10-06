import time

import torch as torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
import math
import torch.utils.model_zoo as model_zoo
# from SemanticResNet50 import ResNetBackbone, Bottleneck
# from models.ResNet50_MLP import ResNetBackbone
# from models.MLP_return_inner_feature import MLPMixer

from functools import partial
from einops.layers.torch import Rearrange, Reduce

# ResNet
from Util.AFF import AFF, DPAFF, DPAFF1
from models.SwinT import swin_tiny_patch4_window7_224, swin_small_patch4_window7_224, swin_base_patch4_window7_224


# 网络权重初始化
def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    # if isinstance(m, nn.Conv2d):
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data)
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


# MLP 预归一化残差块：使用层归一化后输入到fn的结果 再与原始输入相加得到最后输出
class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x


# MLPMixer1：删除Reduce操作
class MLPMixer(nn.Module):
    def __init__(self, image_size, channels, patch_size, dim, depth, expansion_factor=4, dropout=0.):
        """
        Args:
            image_size (int): 输入图像的尺寸 7
            channels (int): 输入图像的通道数 10*64*4 = 2560 (/4=640 *2 = 1280)
            patch_size (int): 每个 patch 的尺寸 1
            dim (int): 隐藏层的维度 64*4 = 256 (/4=64 *2 =128)
            depth (int): 模型的深度，即残差块的数量 9
            expansion_factor (int): 扩展因子，即 MLP 中扩展维度的倍数 4
            dropout (float): dropout 的概率 0
        """
        super().__init__()
        assert (image_size % patch_size) == 0, 'image must be divisible by patch size'
        self.num_patches = (image_size // patch_size) ** 2  # 计算patch数量 7
        self.chan_first, self.chan_last = partial(nn.Conv1d,
                                                  kernel_size=1), nn.Linear  # partial作用是调用 nn.Conv1d 函数并将 kernel_size 参数设置为 1

        self.mlp = nn.Sequential(
            # (batch_size, num_patches, patch_size * patch_size * channels)
            # [b, 4*p*64, 7, 7]即[b, 2560, 7, 7] -> [b, h*w=49, 2560 or 640]
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            # [b, 49, 2560 or 640 or 1280]
            # 映射到隐藏层的维度,nn.Linear只影响最后一维度
            nn.Linear((patch_size ** 2) * channels, dim),  # [b, 49, 2560 or 640 or 1280] -> [b, 49, 64 or 128]
            # 根据深度 构建残差MLP块
            *[nn.Sequential(  # 不会改变维度[b, 49, 64]
                PreNormResidual(dim, self.FeedForward(self.num_patches, expansion_factor, dropout, self.chan_first)),
                PreNormResidual(dim, self.FeedForward(dim, expansion_factor, dropout, self.chan_last))
            ) for _ in range(depth)],
            # 归一化，不改变维度[b, 49, 64]
            nn.LayerNorm(dim),
            # TODO 融合模块0 Reduce换为Rearrange，以满足nn.Cov2d输入的格式
            # Reduce('b n c -> b c', 'mean'),  # 池化 [b, 256] or [b, 64 or 128]
            Rearrange('b (h w) c -> b c h w', h=7, w=7)  # [b, 64 or 128, 7, 7]
            # nn.Linear(dim, num_classes)
        )
        # print(self.mlp)

    def FeedForward(self, dim, expansion_factor=4, dropout=0., dense=nn.Linear):
        return nn.Sequential(
            dense(dim, dim * expansion_factor),
            nn.GELU(),
            nn.Dropout(dropout),
            dense(dim * expansion_factor, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, distillation_layer_num=None):  # 输入的是[b, 4p*64=256p, 7, 7]
        for mlp_single in self.mlp:
            x = mlp_single(x)
        return x


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)
            else:
                pass


# 分数预测回归器
class RegressionFCNet(nn.Module):
    """
    Target network for quality prediction.
    """

    def __init__(self):
        super(RegressionFCNet, self).__init__()
        self.target_in_size = 512
        self.target_fc1_size = 256

        self.sigmoid = nn.Sigmoid()
        self.l1 = nn.Linear(self.target_in_size, self.target_fc1_size)
        self.relu1 = nn.PReLU()
        self.drop1 = nn.Dropout(0.5)
        self.bn1 = nn.BatchNorm1d(256)

        self.l2 = nn.Linear(self.target_fc1_size, 1)

    def forward(self, x):
        q = self.l1(x)
        q = self.l2(q).squeeze()
        return q


# 分数预测回归器
class RegressionFCNet1(nn.Module):
    """
    Target network for quality prediction.
    """

    def __init__(self, in_channel=256):
        super(RegressionFCNet1, self).__init__()
        self.in_channel = in_channel

        self.l1 = nn.Linear(self.in_channel, 1)

    def forward(self, x):
        q = self.l1(x).squeeze()
        return q


# 更换为SwinT
class Student(nn.Module):
    def __init__(self, self_patch_num=10, lda_channel=64, encode_decode_channel=64, MLP_depth=9, distillation_layer=9):
        super(Student, self).__init__()

        self.self_patch_num = self_patch_num
        self.lda_channel = lda_channel
        self.encode_decode_channel = encode_decode_channel
        self.MLP_depth = MLP_depth
        self.distillation_layer_num = distillation_layer

        # TODO 更换backbone 1. 换为SwinT
        # self.feature_extractor = ResNetBackbone()  # 多尺度特征提取器：ResNet50
        # self.feature_extractor = swin_small_patch4_window7_224()  # 多尺度特征提取器
        self.feature_extractor = swin_tiny_patch4_window7_224()  # 多尺度特征提取器
        # TODO 不固定
        for param in self.feature_extractor.parameters():  # 作为特征提取器无需学习
            param.requires_grad = False

        # TODO 更换backbone 2. 处理局部特征的卷积层维度要变
        # 对提取的局部特征的处理：使得输出的特征图的大小都是 (7,7)，通道数都是 self.lda_channel = 64 即[_,64,7,7]
        self.lda1_process = nn.Sequential(nn.Conv2d(192, self.lda_channel, kernel_size=1, stride=1, padding=0),
                                          nn.AdaptiveAvgPool2d((7, 7)))  # 先用1*1卷积核统一通道数为64，再用自适应平均池化统一尺寸为7*7
        self.lda2_process = nn.Sequential(nn.Conv2d(384, self.lda_channel, kernel_size=1, stride=1, padding=0),
                                          nn.AdaptiveAvgPool2d((7, 7)))
        self.lda3_process = nn.Sequential(nn.Conv2d(768, self.lda_channel, kernel_size=1, stride=1, padding=0),
                                          nn.AdaptiveAvgPool2d((7, 7)))
        self.lda4_process = nn.Sequential(nn.Conv2d(768, self.lda_channel, kernel_size=1, stride=1, padding=0),
                                          nn.AdaptiveAvgPool2d((7, 7)))
        self.lda_process = [self.lda1_process, self.lda2_process, self.lda3_process, self.lda4_process]

        self.MLP_encoder_diff = MLPMixer(image_size=7, channels=self.self_patch_num * self.lda_channel,
                                         patch_size=1, dim=self.encode_decode_channel, depth=self.MLP_depth * 2)
        self.MLP_encoder_lq = MLPMixer(image_size=7, channels=self.self_patch_num * self.lda_channel,
                                       patch_size=1, dim=self.encode_decode_channel, depth=self.MLP_depth)
        # 融合模块1 定义
        self.ff = DPAFF1(channels=lda_channel * 4)  # 4层64通道cat在一起

        # 回归器
        self.regressor = RegressionFCNet1()

        # 权重初始化
        initialize_weights(self.MLP_encoder_diff, 0.1)
        initialize_weights(self.MLP_encoder_lq, 0.1)
        initialize_weights(self.regressor, 0.1)

        initialize_weights(self.lda1_process, 0.1)
        initialize_weights(self.lda2_process, 0.1)
        initialize_weights(self.lda3_process, 0.1)
        initialize_weights(self.lda4_process, 0.1)

    def forward(self, LQ_patches, refHQ_patches):
        device = LQ_patches.device
        # 形状处理
        b, p, c, h, w = LQ_patches.shape
        LQ_patches_reshape = LQ_patches.view(b * p, c, h, w)
        refHQ_patches_reshape = refHQ_patches.view(b * p, c, h, w)

        # 分别提取lq和hq图像特征
        # lda1~4的维度：[b*p, 256, 56, 56], [b*p, 512, 28, 28], [b*p, 1024, 14, 14], [b*p, 2048, 7, 7]
        lq_lda_features = self.feature_extractor(LQ_patches_reshape)  # return [lda_1, lda_2, lda_3, lda_4]
        refHQ_lda_features = self.feature_extractor(refHQ_patches_reshape)
        # 依次处理每一层特征，并用一个列表存下
        multi_scale_diff_feature, multi_scale_lq_feature, feature = [], [], []
        for lq_lda_feature, refHQ_lda_feature, lda_process in zip(lq_lda_features, refHQ_lda_features,
                                                                  self.lda_process):
            # 先将四个特征统一为[b*p, 64, 7, 7]再view为[b ,p*64, 7, 7]，p=patch_num=10
            # 由于使用了.permute操作，使得内存不连续，这里改用.reshape方法不要求内存连续
            lq_lda_feature = lda_process(lq_lda_feature).reshape(b, -1, 7, 7)  # view有几个参数就会变形为几个维度
            refHQ_lda_feature = lda_process(refHQ_lda_feature).reshape(b, -1, 7, 7)  # [b ,p*64, 7, 7]
            diff_lda_feature = lq_lda_feature - refHQ_lda_feature  # 差异特征，高质量LQ减去低质量ref

            multi_scale_diff_feature.append(diff_lda_feature)  # 加入多尺度差异特征列表
            multi_scale_lq_feature.append(lq_lda_feature)  # 加入多尺度lq特征列表

        #  4f 四层特征列表分别输入MLP
        lq_mlp_features, diff_mlp_features = [], []
        lq_inner_features, diff_inner_features = [], []
        for lq_feature, diff_fiture in zip(multi_scale_lq_feature, multi_scale_diff_feature):
            # mlp_feature为[b, 64 or 128, 7, 7]，inner_feature为列表，其元素维度为[b, 49, 64 or 128]
            lq_mlp_feature = self.MLP_encoder_lq(lq_feature, self.distillation_layer_num)
            diff_mlp_feature = self.MLP_encoder_diff(diff_fiture, self.distillation_layer_num)
            # lq和diff分别搜集起来mlp和inner特征各四层
            lq_mlp_features.append(lq_mlp_feature)
            diff_mlp_features.append(diff_mlp_feature)

        # 4f 先分别拼接lq和diff的四层mlp特征，再把lq和diff拼接
        lq_mlp_features = torch.cat(lq_mlp_features, dim=1)  # [b, 64*4 = 256 or 512, 7, 7]
        diff_mlp_features = torch.cat(diff_mlp_features, dim=1)  # [b, 64*4 = 256 or 512, 7, 7]

        # 融合模块2 使用
        # feature = torch.cat((lq_mlp_features, diff_mlp_features), 1)  # 拼接两个特征，[b, 256*2 = 512]
        # print(f'lq_mlp_features:{lq_mlp_features.shape}')
        feature = self.ff(diff_mlp_features, lq_mlp_features)  # diff作为qk，lq作为v [b, 256 or 512]
        pred = self.regressor(feature)  # 分数回归
        return pred  # 返回两inner特征列表用于蒸馏L2损失 以及 分数预测用于L1约束

    def _load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))


if __name__ == "__main__":
    # net = ResNetBackbone()
    # x = torch.rand((1, 3, 224, 224))
    # y = net(x)
    # # print(y.shape)
    # for tensor in y:
    #     print(tensor.shape)

    # model = MLPMixer1(image_size=7, channels=2560, patch_size=1, dim=256, depth=18).to('cuda')
    # img = torch.randn(96, 2560, 7, 7).to('cuda')
    # pred1, pred2 = model(img, 18)  # (1, 1000)
    # print(len(pred2))
    # print(len(pred1))
    # print(pred1.shape)

    t1 = time.time()
    m = Student().to('cuda')
    lq = torch.rand((3, 10, 3, 224, 224)).to('cuda')
    hq = torch.rand((3, 10, 3, 224, 224)).to('cuda')
    pred = m(lq, hq)
    print(pred.shape)
    t2 = time.time()
    print(f'运行时间：{t2 - t1}s')
