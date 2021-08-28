import json
import numpy as np
import math
import torch
import torch.nn as nn


def read_dict(path):
    with open(path, encoding='utf-8') as type_dict:
        idx = json.load(type_dict)

    return idx


# sinusoidal embeddings for argument-aware position embeddings
def get_embedding(max_seq_len, embedding_dim, padding_idx=None, rel_pos_init=0):
    """Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    rel pos init:
    """
    # num_embeddings = 2*max_seq_len+1
    num_embeddings = max_seq_len
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float, requires_grad=False) * -emb)
    if rel_pos_init == 0:
        emb = torch.arange(num_embeddings, dtype=torch.float, requires_grad=False).unsqueeze(1) * emb.unsqueeze(0)
    else:
        emb = torch.arange(-max_seq_len,max_seq_len+1, dtype=torch.float, requires_grad=False).unsqueeze(1)*emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
    if embedding_dim % 2 == 1:
        # zero pad
        emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
    if padding_idx is not None:
        emb[padding_idx, :] = 0

    return nn.Embedding.from_pretrained(emb, freeze=True)


class TypeEmbeddings(nn.Module):

    def __init__(self, type_dict_path, embeddinsg_size, is_freeze=False):
        super().__init__()
        self.type2id = read_dict(type_dict_path)
        self.id2type = {v: k for k, v in self.type2id.items()}

        embeddings = [np.random.uniform(-0.25, 0.25, embeddinsg_size) for i in range(len(self.type2id) - 1)]

        embeddings = np.vstack([np.zeros(embeddinsg_size), embeddings])

        self.type_embeddings = nn.Embedding.from_pretrained(torch.FloatTensor(embeddings), freeze=is_freeze)

    def get_type2id(self):
        return self.type2id

    def get_id2type(self):
        return self.id2type

    def forward(self, input_idx):
        return self.type_embeddings(input_idx)


class UtterenceAwarePosEmbedding(nn.Module):

    def __init__(self, max_utter_nums, max_seq_lens, hidden_size, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_utter_nums = max_utter_nums
        self.max_seq_lens = max_seq_lens

        self.utter_embedding = get_embedding(self.max_utter_nums, hidden_size)
        self.linear = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                    nn.Dropout(dropout),
                                    nn.ReLU(inplace=True)
                                    )

    def forward(self, utter_pos, mask):
        utter_pos = self.utter_embedding(utter_pos)

        return self.linear(utter_pos)

