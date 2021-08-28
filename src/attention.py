#!/usr/bin/env python3
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import RelaxedBernoulli


class Gumbel_Selector(nn.Module):

    def __init__(self, input_dim, head_nums, dropout=0.2):
        super().__init__()

        # self.batch_size = batch_size
        self.head_nums = head_nums
        self.dim_per_head = input_dim // head_nums

        self.linear_k = nn.Linear(input_dim, input_dim)
        self.linear_q = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        self.Relu = nn.ReLU()

    def sample_gumbel(self, shape, eps=1e-20):
        U = torch.rand(shape)
        U = U.cuda()
        return -torch.log(-torch.log(U + eps) + eps)

    def forward(self, input, temperature=0.5, is_test=False):
        batch_size = input.size(0)
        query = self.linear_q(input)
        key = self.linear_k(input)

        key = key.reshape(batch_size, -1, self.head_nums, self.dim_per_head).transpose(1, 2).reshape(
             batch_size * self.head_nums, -1, self.dim_per_head)
        query = query.reshape(batch_size, -1, self.head_nums, self.dim_per_head).transpose(1, 2).reshape(
             batch_size * self.head_nums, -1, self.dim_per_head)

        key = key.transpose(1, 2)
        E_s = torch.bmm(query, key)

        if not is_test:
            dist = RelaxedBernoulli(temperature, logits=E_s)
            y = dist.rsample()

        else:
            y = self.sigmoid(E_s)
            y = (y > 0.5).float()

        t_dim = y.size(1)
        y = y.reshape(batch_size, self.head_nums, t_dim, t_dim)

        return y


class ScaleDotProductAttention(nn.Module):
    def __init__(self, model_dim, num_head, attention_dropout=0.2):
        super(ScaleDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

        self.model_dim = model_dim
        self.num_head = num_head

    def forward(self, q, k, v, selector=None, scale=None, attn_mask=None):
        """
        :param q: query [b, l_q, d_q]
        :param k: keys [b, l_k, d_k]
        :param v: values [b, l_v, d_v]， k=v
        :param scale:
        :param attn_mask: masking  [b, l_q, l_k]
        :return:
        """

        attention = torch.bmm(q, k.transpose(1, 2))
        t_dim = attention.size(1)
        if scale:
            attention = attention * scale

        attention = attention.reshape(-1, self.num_head, t_dim, t_dim)
        if selector is not None:
            attention += selector

        if attn_mask is not None:
            attention = attention.masked_fill_(~attn_mask, -1e12)

        attention = attention.reshape(-1, t_dim, t_dim)
        attention = self.softmax(attention)

        attention = self.dropout(attention)
        context = torch.bmm(attention, v)
        return context, attention


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim=768, num_heads=12, dropout=0.2):
        super(MultiHeadAttention, self).__init__()

        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)

        self.dot_product_attention = ScaleDotProductAttention(model_dim, num_heads, dropout)

        self.dropout = nn.Dropout(dropout)

        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, input, selector=None, attn_mask=None, eps=1e-20):

        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = input.size(0)

        key = self.linear_k(input)
        value = self.linear_v(input)
        query = self.linear_q(input)

        # split by heads
        key = key.reshape(batch_size, -1, num_heads, dim_per_head).transpose(1,2).reshape(batch_size*num_heads, -1, dim_per_head)
        value = value.reshape(batch_size, -1, num_heads, dim_per_head).transpose(1,2).reshape(batch_size*num_heads, -1, dim_per_head)
        query = query.reshape(batch_size, -1, num_heads,  dim_per_head).transpose(1,2).reshape(batch_size*num_heads, -1, dim_per_head)

        if selector is not None:
            selector = selector + eps
            selector = torch.log(selector)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1)
            # selector = selector.unsqueeze(1)

        scale = (key.size(-1)) ** -0.5
        context, attention = self.dot_product_attention(
            query, key, value, selector, scale, attn_mask
        )

        # concat heads
        context = context.reshape(batch_size,num_heads, -1,  dim_per_head)
        context = context.transpose(1,2)
        context = context.reshape(batch_size, -1, num_heads*dim_per_head)
        output = context

        return output


class MentionAttention(nn.Module):

    def __init__(self, query_dim, key_dim, dropout=0.2):
        super().__init__()
        self.scale = 1. / math.sqrt(query_dim)
        self.dropout = nn.Dropout(dropout)
        self.W_q = nn.Sequential(
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(query_dim, key_dim, bias=False))

        self.W_k = nn.Sequential(
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(key_dim, key_dim, bias=False))

        self.W_m = nn.Sequential(
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(key_dim, 1, bias=True))

        self.relu = nn.ReLU()

    def forward(self, querys, keys, masks):
        res = []
        for query, key, mask in zip(querys, keys, masks):

            mask = mask.cuda()
            entity_nums = key.size(0)

            query = query.repeat(entity_nums, 1, 1)
            mask = mask.repeat(entity_nums, 1, 1).transpose(0, 1)

            value = key
            query = self.W_q(query)
            w_mention = self.W_m(key).squeeze(2)
            key = self.W_k(key)
            key = key.transpose(1,2)
            attention = torch.bmm(query, key)
            attention = attention.mul_(self.scale)
            attention = attention + w_mention.unsqueeze(1)

            mask = ~mask.bool()
            attention = attention.masked_fill_(mask, -1e12)

            attention = F.softmax(attention, dim=-1)
            attention = self.dropout(attention)
            res.append(torch.bmm(attention, value))

        return res
