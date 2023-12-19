# coding: utf-8

import numpy as np
import math
import copy
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformer import EncoderLayer, PositionEmbedding

class BERT(nn.Module):
    def __init__(self, d_model, d_fc, n_heads, n_layers):
        super(BERT, self).__init__()
        encoder_layer = EncoderLayer(n_heads, d_model, d_fc)
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(n_layers)])

    def forward(self, x, key_padding_mask=None):
        for layer in self.layers:
            x, scores = layer(x, key_padding_mask)
        return x
    
################################################## exp ##################################################
class MyModel(nn.Module):
    def __init__(self, max_len, mlm_cls_num, doc_cls_num, d_model, d_fc, n_heads, n_layers) -> None:
        super(MyModel, self).__init__()
        self.token_embedding = nn.Embedding(mlm_cls_num, d_model)
        self.pos_embedding = PositionEmbedding(max_len, d_model)
        self.bert = BERT(d_model, d_fc, n_heads, n_layers)
        self.mlm_fc = nn.Linear(d_model, mlm_cls_num)
        self.cls_fc = nn.Linear(d_model, doc_cls_num)

    def forward(self, x, key_padding_mask=None):
        x = self.token_embedding(x)
        x = self.pos_embedding(x)
        x = self.bert(x, key_padding_mask)
        yp_mlm = self.mlm_fc(x)
        yp_cls = self.cls_fc(x[:, 0])
        return yp_mlm, yp_cls

def make_pair_data(max_val, max_length, mask_rate=0.2):
    length = random.randint(1, max_length)
    beg = random.randint(0, max_val - length)
    end = beg + length
    x = np.array(list(range(beg, end)))
    y = copy.deepcopy(x)
    for i in range(len(x)):
        if random.random() < mask_rate:
            x[i] = -1
    mask_idxes = (x == -1)
    # 均值的类型：0, 小于中值；1, 大于中值。
    doc_cls = 0 if y.mean() < max_val//2 else 1
    return x, y, mask_idxes, doc_cls

class MyDataSet(Dataset):
    def __init__(self, num_data, max_val, max_length, pad_id=0, mask_id=1, cls_id=2):
        super(MyDataSet, self).__init__()
        self.num_data = num_data
        self.max_val = max_val
        self.max_length = max_length
        self.pad_id = pad_id
        self.mask_id = mask_id
        self.cls_id = cls_id

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        # norm sampling
        x, y, mask_idxes, doc_cls = make_pair_data(self.max_val, self.max_length)

        # add offset: pad/mask/cls
        x += 3
        y += 3

        # reset mask
        x[mask_idxes] = self.mask_id

        # append pad/cls
        x = torch.LongTensor([self.cls_id] + x.tolist() + [self.pad_id] * (self.max_length - len(x)))
        y_mlm = torch.LongTensor([self.pad_id] + y.tolist() + [self.pad_id] * (self.max_length - len(y)))
        key_padding_mask = x.not_equal(self.pad_id)

        y_cls = torch.LongTensor([doc_cls])
        
        return x, y_mlm, key_padding_mask, y_cls
    
if __name__ == '__main__':
    # model configs
    d_model = 256
    d_fc = d_model * 4
    n_heads = 8
    n_layers = 6
    max_length = 15
    vocab = 100
    mlm_cls_num = vocab + 3
    doc_cls_num = 2

    # data configs
    PAD_ID = 0
    MASK_ID = 1
    CLS_ID = 2

    num_data = 100000
    batch_size = 320
    dataset = MyDataSet(num_data, vocab, max_length, PAD_ID, MASK_ID, CLS_ID)
    data_loader = DataLoader(dataset, batch_size = batch_size, shuffle=True, num_workers=0)

    # train configs
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 5
    model = MyModel(max_length + 1, mlm_cls_num, doc_cls_num, d_model, d_fc, n_heads, n_layers)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)

    # train
    model.train()
    for epoch in range(epochs):
        for batch, (x, y_mlm, key_padding_mask, y_cls) in enumerate(data_loader):
            x = x.to(device)
            y_mlm = y_mlm.to(device)
            key_padding_mask = key_padding_mask.to(device)
            y_cls = y_cls.to(device)

            yp_mlm, yp_cls = model(x, key_padding_mask)

            loss = criterion(yp_mlm.view(-1, mlm_cls_num), y_mlm.view(-1)) + criterion(yp_cls.view(-1, doc_cls_num), y_cls.view(-1))
            print(f'epoch: {(epoch + 1)}, batch: {(batch + 1)}, lr: {lr_scheduler.get_last_lr()[0]:.7f}, loss: {loss:.6f}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        lr_scheduler.step()

    # val
    model.eval()
    with torch.no_grad():
        (x, y_mlm, key_padding_mask, y_cls) = next(iter(data_loader))
        x = x.to(device)
        y_mlm = y_mlm.to(device)
        key_padding_mask = key_padding_mask.to(device)
        y_cls = y_cls.to(device)
        
        yp_mlm, yp_cls = model(x, key_padding_mask)

        yp_mlm = F.softmax(yp_mlm, dim = -1)
        yp_mlm = torch.argmax(yp_mlm, dim = -1)
        yp_mlm[y_mlm == PAD_ID] = PAD_ID

        yp_cls = F.softmax(yp_cls, dim = -1)
        yp_cls = torch.argmax(yp_cls, dim = -1)

        print(f'x: {x[0]}')
        print(f'y_mlm: {y_mlm[0]}')
        print(f'yp_mlm: {yp_mlm[0]}')
        print(f'y_cls: {y_cls[0]}')
        print(f'yp_cls: {yp_cls[0]}')