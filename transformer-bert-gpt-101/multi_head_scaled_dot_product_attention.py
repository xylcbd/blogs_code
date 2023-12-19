# coding: utf-8

import math
import random
import torch
import torch.nn.functional as F

from scaled_dot_product_attention import scaled_dot_product_attention

def multi_head_scaled_dot_product_attention(q, k, v, n_heads, q_weight, q_bias, k_weight, k_bias, v_weight, v_bias, out_weight, out_bias, key_padding_mask=None, atten_mask=None):
    # param check
    b, l_q, d = q.shape
    b, l_kv, d = k.shape
    assert q_weight.shape[0] % n_heads == 0
    assert k_weight.shape[0] % n_heads == 0
    assert v_weight.shape[0] % n_heads == 0
    d_header = d // n_heads

    # linear projection
    def _linear(x, w, b):
        r = torch.matmul(x, w.T)
        if b is not None:
            r += b
        return r
    _q = _linear(q, q_weight, q_bias)
    _k = _linear(k, k_weight, k_bias)
    _v = _linear(v, v_weight, v_bias)

    # b, l, d => b, l, n_heads, d_header => b, n_heads, l, d_header => b * n_heads, l, d_header
    _q = _q.view(b, l_q, n_heads, d_header).permute(0, 2, 1, 3).reshape(-1, l_q, d_header)
    _k = _k.view(b, l_kv, n_heads, d_header).permute(0, 2, 1, 3).reshape(-1, l_kv, d_header)
    _v = _v.view(b, l_kv, n_heads, d_header).permute(0, 2, 1, 3).reshape(-1, l_kv, d_header)

    if key_padding_mask is not None:
        assert key_padding_mask.shape[0] == b and key_padding_mask.shape[1] == l_kv
        key_padding_mask = key_padding_mask.view(b, 1, 1, l_kv).expand(-1, n_heads, -1, -1).reshape(b * n_heads, 1, l_kv)
        if atten_mask is None:
            atten_mask = key_padding_mask
        else:
            atten_mask = torch.logical_and(atten_mask, key_padding_mask)

    # scaled dot product attention
    res, score = scaled_dot_product_attention(_q, _k, _v, atten_mask = atten_mask)

    # b * n_heads, l, d_header => b, n_heads, l, d_header => b, l, n_heads, d_header => b, l_q, d
    res = res.view(b, n_heads, l_q, d_header).permute(0, 2, 1, 3).reshape(b, l_q, d)
    res = _linear(res, out_weight, out_bias)

    # b * n_heads, l_q, l_kv => b, n_heads, l_q, l_kv => b, l_q, l_kv
    score = score.view(b, n_heads, l_q, l_kv).mean(dim = 1)

    return res, score

if __name__ == '__main__':
    num_heads = 8
    b, l_q, l_kv, d = 8, 32, 64, 128
    assert d % num_heads == 0
    q = torch.randn((b, l_q, d), dtype=torch.float32)
    k = torch.randn((b, l_kv, d), dtype=torch.float32)
    v = torch.randn((b, l_kv, d), dtype=torch.float32)

    bool_atten_mask = torch.ones(l_q, l_kv, dtype=torch.bool).tril(diagonal=0)
    atten_mask = torch.zeros(bool_atten_mask.shape, dtype=q.dtype)
    atten_mask.masked_fill_(bool_atten_mask.logical_not(), float("-inf"))
    
    bool_key_padding_mask = torch.ones(b, l_kv, dtype=torch.bool)
    for i in range(b):
        pad_len = random.randint(0, l_kv//2)
        bool_key_padding_mask[i, -pad_len:] = False
    key_padding_mask = torch.zeros(bool_key_padding_mask.shape, dtype=q.dtype)
    key_padding_mask.masked_fill_(bool_key_padding_mask.logical_not(), float("-inf"))

    q_weight = torch.randn((d, d), dtype=torch.float32)
    k_weight = torch.randn((d, d), dtype=torch.float32)
    v_weight = torch.randn((d, d), dtype=torch.float32)
    out_weight = torch.randn((d, d), dtype=torch.float32)
    q_bias = torch.randn((d), dtype=torch.float32)
    k_bias = torch.randn((d), dtype=torch.float32)
    v_bias = torch.randn((d), dtype=torch.float32)
    out_bias = torch.randn((d), dtype=torch.float32)

    res, score = multi_head_scaled_dot_product_attention(q, k, v, num_heads, q_weight, q_bias, k_weight, k_bias, v_weight, v_bias, out_weight, out_bias, key_padding_mask = bool_key_padding_mask, atten_mask = bool_atten_mask)

    print(f'num_heads: {num_heads}')
    print(f'q shape: {q.shape}')
    print(f'k shape: {q.shape}')
    print(f'v shape: {q.shape}')
    print(f'q_weight shape: {q_weight.shape}')
    print(f'k_weight shape: {k_weight.shape}')
    print(f'v_weight shape: {v_weight.shape}')
    print(f'out_weight shape: {out_weight.shape}')
    print(f'q_bias shape: {q_bias.shape}')
    print(f'k_bias shape: {k_bias.shape}')
    print(f'v_bias shape: {v_bias.shape}')
    print(f'out_bias shape: {out_bias.shape}')

    print(f'score shape: {score.shape}')
    print(f'res shape: {res.shape}')

    # 与标准pytorch实现进行比对
    _q, _k, _v = (x.transpose(1, 0) for x in (q, k, v))
    _res, _score = F.multi_head_attention_forward(_q, _k, _v, d, num_heads, q_proj_weight=q_weight, k_proj_weight=k_weight, v_proj_weight=v_weight, in_proj_bias=torch.concat([q_bias, k_bias, v_bias], dim=-1), out_proj_weight=out_weight, out_proj_bias=out_bias, key_padding_mask=key_padding_mask, attn_mask=atten_mask, use_separate_proj_weight=True, in_proj_weight=None, bias_k=None, bias_v=None, add_zero_attn=False, dropout_p=0.0, training=False)
    _res = _res.transpose(1, 0)

    print(res[0, 0, :16])
    print(_res[0, 0, :16])