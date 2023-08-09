import numpy as np
import torch
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix
from matplotlib import pyplot as plt

torch.autograd.set_detect_anomaly(True)


def train_class(epoch_num, train_data_set, test_data_set, batch_size=1, net=None, Loss=None, optimizer=None,
                is_use_gpu=True, model_id=None, eval_num=1, log_path='', model_save_path=''):
    if is_use_gpu:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    acc_best = 0
    acc_list = []
    for epoch in range(epoch_num):
        net.train()
        dataloader = DataLoader(dataset=train_data_set, batch_size=batch_size, shuffle=True, drop_last=True,
                                num_workers=0)
        print('start train,the len of dataloader is ', len(dataloader))
        loss_epoch = torch.FloatTensor([0]).to(device)
        for idx, (X, Y) in tqdm(enumerate(dataloader)):
            X = X.to(device)
            Y = Y.to(device)
            out1, out2, out3 = net(X)
            optimizer.zero_grad()
            loss = Loss(out1, Y.view(1, -1)[0]) + Loss(out2, Y.view(1, -1)[0]) + Loss(out3, Y.view(1, -1)[0])
            loss.backward()
            optimizer.step()
            loss_epoch += loss
        print('[epoch{}/{}],loss:{}'.format(epoch, epoch_num, (loss_epoch / len(dataloader)).data[0]))
        print('\n')
        log_file = open(log_path, mode='a+')
        log_file.write('[epoch{}/{}],loss:{}'.format(epoch, epoch_num, (loss_epoch / len(dataloader)).data[0]))
        log_file.write('\n')
        log_file.close()
        if epoch % eval_num == 0:
            acc = test_class(data_set=test_data_set, batch_size=int(batch_size / 2), net=net, is_use_gpu=is_use_gpu,
                             model_id=model_id, log_path=log_path)
            acc_list.append(acc)
            if acc > acc_best:
                acc_best = acc
                torch.save(net.state_dict(), model_save_path)


def test_class(data_set, batch_size=1, net=None, is_use_gpu=True, model_id=None, log_path=''):
    if is_use_gpu:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    net.eval()
    dataloader_eval = DataLoader(dataset=data_set, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=0)
    print('start test,the len of dataloader is ', len(dataloader_eval))
    label_total = []
    predict_total = []
    for idx, (X, Y) in tqdm(enumerate(dataloader_eval)):
        X = X.to(device)
        Y = Y.to(device)
        out1, out2, out3 = net(X)
        max_index = torch.max(out1, dim=1)[1]
        label_total.append(Y.cpu().detach().numpy())
        predict_total.append(max_index.cpu().detach().numpy())
    label_total = np.array(label_total).reshape((1, -1))
    predict_total = np.array(predict_total).reshape((1, -1))
    accuracy = accuracy_score(label_total[0], predict_total[0])
    precision = precision_score(label_total[0], predict_total[0], average='weighted')
    recall = recall_score(label_total[0], predict_total[0], average='weighted')
    f1 = f1_score(label_total[0], predict_total[0], average='weighted')
    con_matrix = confusion_matrix(y_true=label_total[0], y_pred=predict_total[0])
    print('--------------test-----------------------\n')
    print('confusion_matrix\n')
    print(con_matrix)
    print('\n')
    print('[accuracy]:{}\n'.format(accuracy))
    print('[precision]:{}\n'.format(precision))
    print('[recall     ]:{}\n'.format(recall))
    print('[f1-score ]:{}\n'.format(f1))
    print('--------------end test-----------------------\n')
    log_file = open(log_path, mode='a+')
    log_file.write('-------------------------------------\n')
    log_file.write('confusion_matrix\n')
    log_file.write(str(con_matrix))
    log_file.write('\n')
    log_file.write('[accuracy]:{}\n'.format(accuracy))
    log_file.write('[precision]:{}\n'.format(precision))
    log_file.write('[recall     ]:{}\n'.format(recall))
    log_file.write('[f1-score ]:{}\n'.format(f1))
    log_file.write('-------------------------------------\n')
    log_file.close()
    return accuracy
