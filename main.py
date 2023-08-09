import argparse

from torch import nn, optim

from Model import LI_FPN
from Util import train_test_util_binary
import torch

parser = argparse.ArgumentParser('LI_FPN_Arg')
parser.add_argument('--task', type=str, default='depression', help='{depression, anxiety, dep_anx}')
parser.add_argument('--class_num', type=int, default=1)
parser.add_argument('--task_form', type=str, default='r', help='{c for class task, r for rating task}')
parser.add_argument('--lim', type=str, default='g', help='{g for LI_G, s for LI_S}')
parser.add_argument('--backbone', type=str, default='res18', help='{res18, res34, res50}')
parser.add_argument('--len_t', type=int, default=7, help='the length of input')
parser.add_argument('--pretrain', type=bool, default=False)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--model_id', type=str, default='define your model name')
parser.add_argument('--log_path', type=str, default='log path, a txt file')
parser.add_argument('--model_save_path', type=str, default='where the trained model is saved, a folder path')
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
    train_data_set = 'the train dataset'
    test_data_set = 'the test dataset'
    Loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    train_test_util_binary.train_class(epoch_num=args.epoch,
                                       train_data_set=train_data_set,
                                       test_data_set=test_data_set,
                                       batch_size=args.batch_size,
                                       net=model,
                                       Loss=Loss,
                                       optimizer=optimizer,
                                       is_use_gpu=True,
                                       model_id=args.model_id,
                                       eval_num=1,
                                       log_path=args.log_path,
                                       model_save_path=args.model_save_path)
