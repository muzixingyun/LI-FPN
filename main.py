import argparse
from Model import LI_FPN, DC, LIM, STFPN
import torch

parser = argparse.ArgumentParser('LI_FPN_Arg')
parser.add_argument('--task', type=str, default='depression', help='{depression, anxiety, dep_anx}')
parser.add_argument('--class_num', type=int, default=1)
parser.add_argument('--task_form', type=str, default='r', help='{c for class task, r for rating task}')
parser.add_argument('--lim', type=str, default='g', help='{g for LI_G, s for LI_S}')
parser.add_argument('--backbone', type=str, default='res18', help='{res18, res34, res50}')
parser.add_argument('--len_t', type=int, default=7, help='the length of input')
parser.add_argument('--pretrain', type=bool, default=False)
args = parser.parse_args()
print(args)

if __name__ == '__main__':
    device = torch.device('cuda:0')
    model = LI_FPN.LI_FPN(class_num=args.class_num,
                          task_form=args.task_form,
                          lim=args.lim,
                          backbone=args.backbone,
                          len_t=args.len_t,
                          pretrain=args.pretrain).to(device)
    # print(model)
    a = torch.rand((4, 7, 3, 224, 224)).to(device)
    b, c, d = model(a)
    print(b.shape)
    print(c.shape)
    print(d.shape)
