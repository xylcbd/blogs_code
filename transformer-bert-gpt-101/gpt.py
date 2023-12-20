# coding: utf-8

import numpy as np
import copy
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformer import EncoderLayer, PositionEmbedding

class GPT(nn.Module):
    def __init__(self, d_model, d_fc, n_heads, n_layers, max_len, vocab_size):
        super(GPT, self).__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = PositionEmbedding(max_len, d_model)

        layer = EncoderLayer(n_heads, d_model, d_fc)
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x, key_padding_mask=None, atten_mask=None):
        x = self.token_embedding(x)
        x = self.pos_embedding(x)

        for layer in self.layers:
            x, scores = layer(x, key_padding_mask, atten_mask)
        
        x = self.fc(x)

        return x
    
################################################## exp ##################################################
def generate(gpt_model, token_idx, max_seq_size):
    pass

def make_pair_data(max_val, max_length):
    length = random.randint(1, max_length)
    beg = random.randint(0, max_val - length)
    end = beg + length
    seq = np.array(list(range(beg, end)))
    return seq

class MyDataSet(Dataset):
    def __init__(self, num_data, max_val, max_length, pad_id=0, bos_id=1, eos_id=2):
        super(MyDataSet, self).__init__()
        self.num_data = num_data
        self.max_val = max_val
        self.max_length = max_length
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        # 给定一个起始数字，如果该数字大于阈值，则输出EOS结束，否则输出这个数字的下一个数，直到当前数大于阈值
        assert self.max_length > 1

        # norm sampling
        inp_seq = make_pair_data(self.max_val, self.max_length - 1)
        # print(f'raw inp_seq: {inp_seq}')

        # add offset: pad/bos/eos
        inp_seq += 3
        # print(f'offset inp_seq: {inp_seq}')

        # add bos and shift
        out_seq = inp_seq.tolist() + [inp_seq[-1]+1]
        inp_seq = [self.bos_id] + inp_seq.tolist()
        # print(f'shift inp_seq: {inp_seq}')
        # print(f'shift out_seq: {out_seq}')

        # update eos/pad
        thre = (self.max_val // 2 + 3)
        # print(f'thre: {thre}')
        eos_idx = None
        for i in range(len(inp_seq)):
            if eos_idx is None and inp_seq[i] >= thre:
                eos_idx = i
            if eos_idx is not None:
                if i == eos_idx:
                    out_seq[i] = self.eos_id
                    continue
                else:
                    inp_seq[i] = self.pad_id
                    out_seq[i] = self.pad_id
        # print(f'update inp_seq: {inp_seq}')
        # print(f'update out_seq: {out_seq}')

        assert len(inp_seq) == len(out_seq)
        inp_seq = torch.LongTensor(inp_seq + [self.pad_id] * (self.max_length - len(inp_seq)))
        out_seq = torch.LongTensor(out_seq + [self.pad_id] * (self.max_length - len(out_seq)))
        
        key_padding_mask = inp_seq.not_equal(self.pad_id)
        length = inp_seq.shape[0]
        atten_mask = torch.ones(length, length, dtype=torch.bool).tril(diagonal=0)

        # print(f'final inp_seq: {inp_seq}')
        # print(f'final out_seq: {out_seq}')

        # print(f'final key_padding_mask: {key_padding_mask}')
        # print(f'final atten_mask: {atten_mask}')

        return inp_seq, out_seq, key_padding_mask, atten_mask
    
if __name__ == '__main__':
    # model configs
    d_model = 256
    d_fc = d_model * 4
    n_heads = 8
    n_layers = 6
    max_length = 16
    vocab_size = 100
    cls_num = vocab_size + 3

    # data configs
    PAD_ID = 0
    BOS_ID = 1
    EOS_ID = 2

    num_data = 100000
    batch_size = 320
    dataset = MyDataSet(num_data, vocab_size, max_length, PAD_ID, BOS_ID, EOS_ID)
    data_loader = DataLoader(dataset, batch_size = batch_size, shuffle=True, num_workers=0)

    # train configs
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 5
    model = GPT(d_model, d_fc, n_heads, n_layers, max_length, cls_num)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)

    # train
    model.train()
    for epoch in range(epochs):
        for batch, (inp_seq, out_seq, key_padding_mask, atten_mask) in enumerate(data_loader):
            inp_seq = inp_seq.to(device)
            out_seq = out_seq.to(device)
            key_padding_mask = key_padding_mask.to(device)
            atten_mask = atten_mask.to(device)

            p_seq = model(inp_seq, key_padding_mask, atten_mask)

            # ignore first token
            loss = criterion(p_seq[:, 1:].reshape(-1, cls_num), out_seq[:, 1:].reshape(-1))
            print(f'epoch: {(epoch + 1)}, batch: {(batch + 1)}, lr: {lr_scheduler.get_last_lr()[0]:.7f}, loss: {loss:.6f}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        lr_scheduler.step()

    # val
    model.eval()
    with torch.no_grad():
        (inp_seq, out_seq, key_padding_mask, atten_mask) = next(iter(data_loader))
        inp_seq = inp_seq.to(device)
        out_seq = out_seq.to(device)
        key_padding_mask = key_padding_mask.to(device)
        atten_mask = atten_mask.to(device)

        p_seq = model(inp_seq, key_padding_mask, atten_mask)

        p_seq = F.softmax(p_seq, dim = -1)
        p_seq = torch.argmax(p_seq, dim = -1)
        p_seq[out_seq == PAD_ID] = PAD_ID
        
        # ignore first token
        p_seq[:][0] = PAD_ID

        print(f'inp_seq: {inp_seq[0]}')
        print(f'out_seq: {out_seq[0]}')
        print(f'p_seq: {p_seq[0]}')