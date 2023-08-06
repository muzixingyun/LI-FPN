import torch
from torch import nn


class FPN(nn.Module):
    def __init__(self, channel_list=[64, 128, 256, 512], class_num=2):
        super(FPN, self).__init__()
        self.class_num = class_num
        self.classfiers = nn.ModuleList([])
        for i in range(len(channel_list)):
            self.classfiers.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(1),
                    nn.Linear(channel_list[i], class_num)
                )
            )
        self.downsample1 = nn.Sequential(
            nn.Conv2d(channel_list[0], channel_list[1], kernel_size=1, padding=0, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            nn.GELU()
        )
        self.downsample2 = nn.Sequential(
            nn.Conv2d(channel_list[1], channel_list[2], kernel_size=1, padding=0, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            nn.GELU()
        )
        self.downsample3 = nn.Sequential(
            nn.Conv2d(channel_list[2], channel_list[3], kernel_size=1, padding=0, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            nn.GELU()
        )
        self.classfier = nn.Sequential(
            nn.Linear(512, class_num)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flt = nn.Flatten()

    def forward(self, feature_list, b, t):
        # 重置传入图像的形状
        out_feature_maps = []
        # 获取每次输入图像的输出数据
        for i in range(len(feature_list)):
            b_t, c, h, w = feature_list[i].shape
            feature_list[i] = torch.sum(feature_list[i].reshape((b, t, c, h, w)), dim=1)
            out_map = self.classfiers[i](feature_list[i])
            out_feature_maps.append(out_map)
        out_feature_maps = torch.stack(out_feature_maps, dim=0)
        out_sfpn = torch.sum(out_feature_maps.transpose(0, 1), dim=1)
        out_feature_maps = torch.softmax(out_feature_maps, dim=-1)
        base_feater_maps = out_feature_maps[:, :, 0].unsqueeze(-1).expand(out_feature_maps.shape)
        dif = torch.abs(out_feature_maps - base_feater_maps)
        dif = torch.sum(dif, dim=-1)
        # if self.class_num == 2:
        #     dif = torch.abs(out_feature_maps[:, :, 0] - out_feature_maps[:, :, 1])  # Layer b
        # else:
        #     dif_one = torch.abs(out_feature_maps[:, :, 0] - out_feature_maps[:, :, 1])  # Layer b
        #     dif_two = torch.abs(out_feature_maps[:, :, 2] - out_feature_maps[:, :, 1])  # Layer b
        #     dif = dif_one + dif_two
        # dif = dif.cuda()
        dif = torch.softmax(dif, dim=0)
        for i in range(len(feature_list)):
            dif_lay = dif[i].unsqueeze(1).unsqueeze(1).unsqueeze(1)
            feature_list[i] = feature_list[i] * dif_lay

        out_fusion1 = self.downsample1(feature_list[0]) + feature_list[1]
        out_fusion2 = self.downsample2(out_fusion1) + feature_list[2]
        out_fusion3 = self.downsample3(out_fusion2) + feature_list[3]
        out_avg = self.avgpool(out_fusion3)
        out_flt = self.flt(out_avg)
        out = self.classfier(out_flt)
        return out, out_sfpn


class FPN_T(nn.Module):
    def __init__(self, b, t, c, class_num=2):
        super(FPN_T, self).__init__()
        self.class_num = class_num
        # self.Transformers = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Flatten(2),
        #     PositionalEncoding(d_model=c, dropout=0.2),
        #     Transformer(feather_len=c, num_layers=4)
        # )
        self.classfiers = nn.ModuleList([])
        for i in range(t):
            self.classfiers.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(1),
                    nn.Linear(c, class_num)
                )
            )

    def forward(self, feature, b, t):
        # 重置传入图像的形状 b_t,c,h,w -》 b,t,c,h,w
        b_t, c, h, w = feature.shape
        feature = feature.reshape(b, t, c, h, w)
        # out_transformer = self.Transformers(feature)
        out_decisions = []
        for i in range(t):
            out_decision = self.classfiers[i](feature[:, i, ::])
            out_decisions.append(out_decision)
        out_decisions = torch.stack(out_decisions, dim=1)
        out_sfpn_t = torch.sum(out_decisions, dim=1)
        out_decisions = torch.softmax(out_decisions, dim=-1)
        base_feater_maps = out_decisions[:, :, 0].unsqueeze(-1).expand(out_decisions.shape)
        dif = torch.abs(out_decisions - base_feater_maps)
        dif = torch.sum(dif, dim=-1)
        dif = torch.softmax(dif, dim=-1)
        dif = dif.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return dif.reshape(b_t, 1, 1, 1), out_sfpn_t


class FPN_R(nn.Module):
    def __init__(self, channel_list=[64, 128, 256, 512], class_num=2):
        super(FPN_R, self).__init__()
        self.class_num = class_num
        self.classfiers = nn.ModuleList([])
        for i in range(len(channel_list)):
            self.classfiers.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(1),
                    nn.Linear(channel_list[i], class_num)
                )
            )
        self.downsample1 = nn.Sequential(
            nn.Conv2d(channel_list[0], channel_list[1], kernel_size=1, padding=0, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            nn.GELU()
        )
        self.downsample2 = nn.Sequential(
            nn.Conv2d(channel_list[1], channel_list[2], kernel_size=1, padding=0, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            nn.GELU()
        )
        self.downsample3 = nn.Sequential(
            nn.Conv2d(channel_list[2], channel_list[3], kernel_size=1, padding=0, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            nn.GELU()
        )
        self.classfier = nn.Sequential(
            nn.Linear(512, class_num)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flt = nn.Flatten()

    def forward(self, feature_list, b, t):
        # 重置传入图像的形状
        out_feature_maps = []
        # 获取每次输入图像的输出数据
        for i in range(len(feature_list)):
            b_t, c, h, w = feature_list[i].shape
            feature_list[i] = torch.sum(feature_list[i].reshape((b, t, c, h, w)), dim=1)
            out_map = self.classfiers[i](feature_list[i])
            out_feature_maps.append(out_map)

        out_feature_maps = torch.stack(out_feature_maps, dim=0)
        out_sfpn = torch.mean(out_feature_maps.transpose(0, 1), dim=1)
        # out_feature_maps = torch.softmax(out_feature_maps, dim=-1)
        # print(out_feature_maps.shape)
        # base_feater_maps = out_feature_maps[:, :, 0].unsqueeze(-1).expand(out_feature_maps.shape)
        base_feater_maps = torch.mean(out_feature_maps,dim=0).unsqueeze(0).expand(out_feature_maps.shape)
        # print(base_feater_maps.shape)
        dif = torch.abs(out_feature_maps - base_feater_maps)
        dif = torch.sum(dif, dim=-1)
        # if self.class_num == 2:
        #     dif = torch.abs(out_feature_maps[:, :, 0] - out_feature_maps[:, :, 1])  # Layer b
        # else:
        #     dif_one = torch.abs(out_feature_maps[:, :, 0] - out_feature_maps[:, :, 1])  # Layer b
        #     dif_two = torch.abs(out_feature_maps[:, :, 2] - out_feature_maps[:, :, 1])  # Layer b
        #     dif = dif_one + dif_two
        # dif = dif.cuda()
        dif = torch.softmax(dif, dim=0)
        for i in range(len(feature_list)):
            dif_lay = dif[i].unsqueeze(1).unsqueeze(1).unsqueeze(1)
            feature_list[i] = feature_list[i] * dif_lay

        out_fusion1 = self.downsample1(feature_list[0]) + feature_list[1]
        out_fusion2 = self.downsample2(out_fusion1) + feature_list[2]
        out_fusion3 = self.downsample3(out_fusion2) + feature_list[3]
        out_avg = self.avgpool(out_fusion3)
        out_flt = self.flt(out_avg)
        out = self.classfier(out_flt)
        return out, out_sfpn


class FPN_R_T(nn.Module):
    def __init__(self, b, t, c, class_num=2):
        super(FPN_R_T, self).__init__()
        self.class_num = class_num
        # self.Transformers = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Flatten(2),
        #     PositionalEncoding(d_model=c, dropout=0.2),
        #     Transformer(feather_len=c, num_layers=4)
        # )
        self.classfiers = nn.ModuleList([])
        for i in range(t):
            self.classfiers.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(1),
                    nn.Linear(c, class_num)
                )
            )

    def forward(self, feature, b, t):
        # 重置传入图像的形状 b_t,c,h,w -》 b,t,c,h,w
        b_t, c, h, w = feature.shape
        feature = feature.reshape(b, t, c, h, w)
        # out_transformer = self.Transformers(feature)

        out_decisions = []
        for i in range(t):
            out_decision = self.classfiers[i](feature[:, i, ::])
            out_decisions.append(out_decision)

        out_decisions = torch.stack(out_decisions, dim=1)
        out_sfpn_t = torch.mean(out_decisions, dim=1)
        # out_decisions = torch.softmax(out_decisions, dim=-1)
        base_feater_maps = torch.mean(out_decisions, dim=1).unsqueeze(1).expand(out_decisions.shape)
        # print(base_feater_maps)
        # base_feater_maps = out_decisions[:, :, 0].unsqueeze(-1).expand(out_decisions.shape)
        dif = torch.abs(out_decisions - base_feater_maps)
        dif = torch.sum(dif, dim=-1)
        dif = torch.softmax(dif, dim=-1)
        dif = dif.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return dif.reshape(b_t, 1, 1, 1), out_sfpn_t

