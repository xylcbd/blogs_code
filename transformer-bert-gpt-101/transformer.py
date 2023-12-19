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
from torch.nn.init import xavier_normal_, xavier_uniform_

from multi_head_scaled_dot_product_attention import multi_head_scaled_dot_product_attention

class PositionEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super(PositionEmbedding, self).__init__()

        self.max_len = max_len
        self.d_model = d_model
        self.pe = torch.zeros(self.max_len, self.d_model, dtype = torch.float32)
        pos = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0)/self.d_model))
        self.pe[:, 0::2] = torch.sin(pos * div_term)
        self.pe[:, 1::2] = torch.cos(pos * div_term)

    def forward(self, x):
        b, l, d = x.shape
        assert d == self.d_model
        assert l <= self.max_len
        return x + self.pe[:l, :].to(x.device).expand_as(x).clone().detach()

class MultiheadAttention(nn.Module):
    def __init__(self, n_heads, d_model):
        super(MultiheadAttention, self).__init__()

        self.n_heads = n_heads
        self.d_model = d_model
        self.q_weight = nn.Parameter(torch.empty((d_model, d_model), dtype=torch.float32), requires_grad=True)
        self.k_weight = nn.Parameter(torch.empty((d_model, d_model), dtype=torch.float32), requires_grad=True)
        self.v_weight = nn.Parameter(torch.empty((d_model, d_model), dtype=torch.float32), requires_grad=True)
        self.out_weight = nn.Parameter(torch.empty((d_model, d_model), dtype=torch.float32), requires_grad=True)
        self.q_bias = nn.Parameter(torch.empty((1, 1, d_model), dtype=torch.float32), requires_grad=True)
        self.k_bias = nn.Parameter(torch.empty((1, 1, d_model), dtype=torch.float32), requires_grad=True)
        self.v_bias = nn.Parameter(torch.empty((1, 1, d_model), dtype=torch.float32), requires_grad=True)
        self.out_bias = nn.Parameter(torch.empty((1, 1, d_model), dtype=torch.float32), requires_grad=True)

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.q_weight)
        xavier_uniform_(self.k_weight)
        xavier_uniform_(self.v_weight)
        xavier_uniform_(self.out_weight)
        xavier_normal_(self.q_bias)
        xavier_normal_(self.k_bias)
        xavier_normal_(self.v_bias)
        xavier_normal_(self.out_bias)

    def forward(self, q, k, v, key_padding_mask = None, atten_mask = None):
        res, score = multi_head_scaled_dot_product_attention(q, k, v, self.n_heads, self.q_weight, self.q_bias, self.k_weight, self.k_bias, self.v_weight, self.v_bias, self.out_weight, self.out_bias, key_padding_mask=key_padding_mask, atten_mask=atten_mask)
        return res, score

class EncoderLayer(nn.Module):
    def __init__(self, n_heads, d_model, d_fc):
        super(EncoderLayer, self).__init__()
        self.self_mhsa = MultiheadAttention(n_heads, d_model)
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_fc, bias=False),
            nn.ReLU(),
            nn.Linear(d_fc, d_model, bias=False)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, key_padding_mask = None, atten_mask = None):
        # post norm
        res, score = self.self_mhsa(x, x, x, key_padding_mask = key_padding_mask, atten_mask = atten_mask)
        res = self.norm1(x + res)
        res = self.norm2(x + self.fc(res))
        return res, score

class DecoderLayer(nn.Module):
    def __init__(self, n_heads, d_model, d_fc):
        super().__init__()
        self.n_heads = n_heads
        self.self_atten = MultiheadAttention(n_heads, d_model)
        self.cross_atten = MultiheadAttention(n_heads, d_model)

        self.fc = nn.Sequential(
            nn.Linear(d_model, d_fc, bias=False),
            nn.ReLU(),
            nn.Linear(d_fc, d_model, bias=False)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, y, memory, y_key_padding_mask=None, self_atten_mask=None, memory_key_padding_mask=None, cross_atten_mask=None):
        res1, self_score = self.self_atten(y, y, y, key_padding_mask = y_key_padding_mask, atten_mask = self_atten_mask)
        res1 = self.norm1(y + res1)

        res2, cross_score = self.cross_atten(res1, memory, memory, key_padding_mask = memory_key_padding_mask, atten_mask = cross_atten_mask)
        res2 = self.norm2(res1 + res2)

        res3 = self.norm3(res2 + self.fc(res2))
        
        return res3, self_score, cross_score

class Encoder(nn.Module):
    def __init__(self, n_layers, encoder_layer):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(n_layers)])

    def forward(self, x, key_padding_mask=None, atten_mask=None):
        scores = []
        for layer in self.layers:
            x, score = layer(x, key_padding_mask=key_padding_mask, atten_mask=atten_mask)
            scores.append(score)
        return x, scores

class Decoder(nn.Module):
    def __init__(self, n_layers, decoder_layer):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(n_layers)])

    def forward(self, y, memory, key_padding_mask=None, self_atten_mask=None, memory_key_padding_mask=None, cross_atten_mask=None):
        self_scores = []
        cross_scores = []
        for layer in self.layers:
            y, self_score, cross_score = layer(y, memory, key_padding_mask, self_atten_mask, memory_key_padding_mask, cross_atten_mask)
            self_scores.append(self_score)
            cross_scores.append(cross_score)
        return y, self_scores, cross_scores

class Transformer(nn.Module):
    def __init__(self, d_model, d_fc, n_heads, n_encoder_layers, n_decoder_layers):
        super(Transformer, self).__init__()
        encoder_layer = EncoderLayer(n_heads, d_model, d_fc)
        self.encoder = Encoder(n_encoder_layers, encoder_layer)
        decoder_layer = DecoderLayer(n_heads, d_model, d_fc)
        self.decoder = Decoder(n_decoder_layers, decoder_layer)

    def forward(self, x, y, x_key_padding_mask=None, x_self_atten_mask=None, y_key_padding_mask=None, y_self_atten_mask=None, y_mem_key_padding_mask=None, y_cross_atten_mask=None):
        memory, x_self_scores = self.encoder(x, x_key_padding_mask, x_self_atten_mask)
        
        y, y_self_scores, y_cross_scores = self.decoder(y, memory, y_key_padding_mask, y_self_atten_mask, y_mem_key_padding_mask, y_cross_atten_mask)
        
        return memory, y, [x_self_scores, y_self_scores, y_cross_scores]
    
################################################## exp ##################################################
class MyModel(nn.Module):
    def __init__(self, max_len, x_vocab, y_vocab, d_model, d_fc, n_heads, n_encoder_layers, n_decoder_layers) -> None:
        super(MyModel, self).__init__()
        self.x_embedding = nn.Embedding(x_vocab, d_model)
        self.y_embedding = nn.Embedding(y_vocab, d_model)
        self.transformer = Transformer(d_model, d_fc, n_heads, n_encoder_layers, n_decoder_layers)
        self.pe = PositionEmbedding(max_len, d_model)
        self.fc = nn.Linear(d_model, y_vocab)

    def forward(self, x, y, x_key_padding_mask=None, y_key_padding_mask=None, y_self_atten_mask=None, y_mem_key_padding_mask=None):
        x = self.x_embedding(x)
        x = self.pe(x)

        y = self.y_embedding(y)
        y = self.pe(y)

        x, y, attens = self.transformer(x, y, x_key_padding_mask, None, y_key_padding_mask, y_self_atten_mask, y_mem_key_padding_mask, None)

        y = self.fc(y)

        return y

def make_pair_data(nums, max_length):
    length = random.randint(1, max_length)
    x = np.random.choice(nums, length)
    y = np.zeros(x.shape, dtype=x.dtype)
    for i in range(len(x)):
        new_order = 0
        cur = x[i]
        for j in range(i):
            if cur >= x[j]:
                new_order += 1
        y[i] = new_order
    return x, y

class MyDataSet(Dataset):
    def __init__(self, num_data, max_val, max_length, pad_id, bos_id, eos_id):
        super(MyDataSet, self).__init__()
        self.num_data = num_data
        self.max_val = max_val
        self.max_length = max_length
        self.nums = range(max_val)
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        # norm sampling
        x, y = make_pair_data(self.nums, self.max_length)
        
        # add offset: pad/bos/eos
        x += 3
        y += 3

        # append pad/bos/eos
        x = torch.LongTensor(x.tolist() + [self.pad_id] * (self.max_length - len(x)))
        y_inp = torch.LongTensor([self.bos_id] + y.tolist() + [self.pad_id] * (self.max_length - len(y)))
        y_out = torch.LongTensor(y.tolist() + [self.eos_id] + [self.pad_id] * (self.max_length - len(y)))

        x_key_padding_mask = x.not_equal(self.pad_id)
        y_key_padding_mask = y_inp.not_equal(self.pad_id)
        y_length = y_inp.shape[0]
        y_self_atten_mask = torch.ones(y_length, y_length, dtype=torch.bool).tril(diagonal=0)
        y_mem_key_padding_mask = x.not_equal(self.pad_id)

        return x, y_inp, y_out, x_key_padding_mask, y_key_padding_mask, y_self_atten_mask, y_mem_key_padding_mask

if __name__ == '__main__':
    # model configs
    d_model = 256
    d_fc = d_model * 4
    n_heads = 8
    n_encoder_layers = 6
    n_decoder_layers = 6
    max_length = 6
    x_vocab = 100
    y_vocab = max_length

    # data configs
    PAD_ID = 0
    BOS_ID = 1
    EOS_ID = 2

    num_data = 100000
    batch_size = 320
    dataset = MyDataSet(num_data, x_vocab, max_length, PAD_ID, BOS_ID, EOS_ID)
    data_loader = DataLoader(dataset, batch_size = batch_size, shuffle=True, num_workers=0)

    # train configs
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 5
    model = MyModel(max_length + 1, x_vocab + 3, y_vocab + 3, d_model, d_fc, n_heads, n_encoder_layers, n_decoder_layers)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)

    # train
    model.train()
    for epoch in range(epochs):
        for batch, (x, y_inp, y_out, x_key_padding_mask, y_key_padding_mask, y_self_atten_mask, y_mem_key_padding_mask) in enumerate(data_loader):
            x = x.to(device)
            y_inp = y_inp.to(device)
            y_out = y_out.to(device)
            x_key_padding_mask = x_key_padding_mask.to(device)
            y_key_padding_mask = y_key_padding_mask.to(device)
            y_self_atten_mask = y_self_atten_mask.to(device)[0]
            y_mem_key_padding_mask = y_mem_key_padding_mask.to(device)
            yp = model(x, y_inp, x_key_padding_mask, y_key_padding_mask, y_self_atten_mask, y_mem_key_padding_mask)

            loss = criterion(yp.view(-1, y_vocab + 3), y_out.view(-1))
            print(f'epoch: {(epoch + 1)}, batch: {(batch + 1)}, lr: {lr_scheduler.get_last_lr()[0]:.7f}, loss: {loss:.6f}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        lr_scheduler.step()

    # val
    model.eval()
    with torch.no_grad():
        (x, y_inp, y_out, x_key_padding_mask, y_key_padding_mask, y_self_atten_mask, y_mem_key_padding_mask) = next(iter(data_loader))
        x = x.to(device)
        y_inp = y_inp.to(device)
        y_out = y_out.to(device)
        x_key_padding_mask = x_key_padding_mask.to(device)
        y_key_padding_mask = y_key_padding_mask.to(device)
        y_self_atten_mask = y_self_atten_mask.to(device)[0]
        y_mem_key_padding_mask = y_mem_key_padding_mask.to(device)
        yp = model(x, y_inp, x_key_padding_mask, y_key_padding_mask, y_self_atten_mask, y_mem_key_padding_mask)
        yp = F.softmax(yp, dim = -1)
        ypg = torch.argmax(yp, dim = -1)
        ypg[y_out == PAD_ID] = PAD_ID
        
        print(f'x: {x[0]}')
        print(f'y_inp: {y_inp[0]}')
        print(f'y_out: {y_out[0]}')
        print(f'ypg: {ypg[0]}')