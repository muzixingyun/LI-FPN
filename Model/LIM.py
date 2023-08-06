import torch
from torch import nn

class LIM_G(nn.Module):
    def __init__(self, in_channel):
        super(LIM_G, self).__init__()
        self.l_conv = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1, stride=1),
            nn.Sigmoid()
        )

    def forward(self, x, b, t):
        b_t, c, h, w = x.shape
        x = x.reshape(b, t, c, h, w)
        x_avg = torch.sum(x, dim=1)  # 将所有特征相加
        # 通过卷积获取特征图
        att_map = self.l_conv(x_avg)
        att_map = att_map.unsqueeze(1).expand(x.shape)
        att_feature = (att_map * x).reshape(b_t, c, h, w)
        return att_feature


class LIM_S(nn.Module):
    def __init__(self, in_channel, t):
        super(LIM_S, self).__init__()
        self.l_conv = nn.ModuleList([])
        for i in range(t):
            self.l_conv.append(
                nn.Sequential(
                    nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1, stride=1),
                    nn.Sigmoid()
                )
            )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, b, t):
        b_t, c, h, w = x.shape
        x = x.reshape(b, t, c, h, w)
        # 通过卷积获取特征图
        summary_all = []
        for i in range(t):
            summary_t = self.l_conv[i](x[:, i, ::])  # 4,c,h,w
            summary_all.append(summary_t)  # 4,c,h,w
        summary_all = torch.stack(summary_all, dim=1)  # 4,t,c,h,w
        community = self.sigmoid(torch.sum(summary_all, dim=1))  # 4,c,h,w
        att_map = community.unsqueeze(1).expand(x.shape)  # 4,1,c,h,w -> 4,t,c,h,w
        att_feature = (att_map * x).reshape(b_t, c, h, w)
        return att_feature

