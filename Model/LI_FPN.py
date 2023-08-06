from torch import nn
from torchvision.models import resnet18, resnet34, resnet50
from Model import LIM
from Model import DC
from Model import STFPN


class LI_FPN(nn.Module):
    def __init__(self, class_num, task_form, lim, backbone, len_t, pretrain):
        super(LI_FPN, self).__init__()

        if task_form == 'r':
            class_num = 1

        if backbone == 'res18':
            self.backbone = resnet18(pretrained=pretrain)
        elif backbone == 'res34':
            self.backbone = resnet34(pretrained=pretrain)
        elif backbone == 'res50':
            self.backbone = resnet50(pretrained=pretrain)

        if lim == 'g':
            self.LIM1 = LIM.LIM_G(in_channel=64)
            self.LIM2 = LIM.LIM_G(in_channel=128)
            self.LIM3 = LIM.LIM_G(in_channel=256)
            self.LIM4 = LIM.LIM_G(in_channel=512)
        elif lim == 's':
            self.LIM1 = LIM.LIM_S(in_channel=64, t=len_t)
            self.LIM2 = LIM.LIM_S(in_channel=128, t=len_t)
            self.LIM3 = LIM.LIM_S(in_channel=256, t=len_t)
            self.LIM4 = LIM.LIM_S(in_channel=512, t=len_t)
        self.DC = DC.Dence_Connect(channel_list=[64, 128, 256, 512])
        if task_form == 'c':
            self.SFPN_T = STFPN.FPN_T(b=4, t=7, c=512, class_num=class_num)
            self.SFPN = STFPN.FPN(channel_list=[64, 128, 256, 512], class_num=class_num)
        elif task_form == 'r':
            self.SFPN_T = STFPN.FPN_R_T(b=4, t=7, c=512, class_num=class_num)
            self.SFPN = STFPN.FPN_R(channel_list=[64, 128, 256, 512], class_num=class_num)

    def forward(self, x):
        # 重置传入图像的形状
        b, t, c, h, w = x.shape
        x = x.reshape((b * t, c, h, w))
        # 图像通过前置卷积结构
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        # 特征通过第一层
        out_layer1 = self.backbone.layer1(x)
        # 特征通过学习和模仿层
        out_lim1 = self.LIM1(out_layer1, b, t)
        # 特征通过第二层
        out_layer2 = self.backbone.layer2(out_lim1)
        # 特征通过学习和模仿层
        out_lim2 = self.LIM2(out_layer2, b, t)
        # 特征通过第三层
        out_layer3 = self.backbone.layer3(out_lim2)
        # 特征通过学习和模仿层
        out_lim3 = self.LIM3(out_layer3, b, t)
        # 特征通过第四层
        out_layer4 = self.backbone.layer4(out_lim3)
        # 特征通过学习和模仿层
        out_lim4 = self.LIM4(out_layer4, b, t)
        # b_t,c,h,w
        out_dc = self.DC([out_lim1, out_lim2, out_lim3, out_lim4])
        out_sfpn_t, decision_sfpn_t = self.SFPN_T(out_dc, b, t)
        out, decision_sfpn = self.SFPN([
            out_lim1 * out_sfpn_t, out_lim2 * out_sfpn_t, out_lim3 * out_sfpn_t, out_lim4 * out_sfpn_t
        ], b, t)
        return out, decision_sfpn, decision_sfpn_t
        # return out
