# coding: utf-8

import math
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(q, k, v, atten_mask=None):
    # param check
    assert q.shape[0] == k.shape[0] and q.shape[0] == v.shape[0] and q.shape[-1] == k.shape[-1] and q.shape[-1] == v.shape[-1]
    b, l_q, d = q.shape
    b, l_kv, d = k.shape
    # bld -> bdl
    kt = k.permute(0, 2, 1)
    logits = torch.matmul(q, kt) / math.sqrt(d)
    bias = torch.zeros(logits.shape, dtype=q.dtype, device=logits.device)
    if atten_mask is not None:
        bias.masked_fill_(atten_mask.expand_as(logits).logical_not(), float("-inf"))
    logits += bias
    score = F.softmax(logits, dim = -1)
    res = torch.matmul(score, v)
    return res, score

if __name__ == '__main__':
    b, l_q, l_kv, d = 8, 32, 64, 128
    q = torch.randn((b, l_q, d), dtype=torch.float32)
    k = torch.randn((b, l_kv, d), dtype=torch.float32)
    v = torch.randn((b, l_kv, d), dtype=torch.float32)
    atten_mask = torch.ones(l_q, l_kv, dtype=torch.bool).tril(diagonal=0)
    res, score = scaled_dot_product_attention(q, k, v, atten_mask=atten_mask)

    print(f'q shape: {q.shape}')
    print(f'k shape: {q.shape}')
    print(f'v shape: {q.shape}')
    print(f'score shape: {score.shape}')
    print(f'res shape: {res.shape}')

    # 与标准pytorch实现进行比对
    _res = F.scaled_dot_product_attention(q, k, v, attn_mask=atten_mask)
    print(res[0, 0, :16])
    print(_res[0, 0, :16])