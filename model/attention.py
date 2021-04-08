import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from model.layers import clones


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    ## (Encode:) query, key are both [30, 8, 10, 64], scores is [30, 8, 10, 10]
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9) # sets masked scores to (almost) -inf
    p_attn = F.softmax(scores, dim = -1) # computes softmax along last dimension
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4) # input size = 512, output size = 512
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            # For Encoder layers, mask is shape [30, 1, 1, 10]
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k

        ## applies linear transformation to query, key, values
        ## then reshapes to [30, -1, 8, 512//8], and transposes to [30, 8, -1, 512//8] (-1 is 9 or 10)
        ## the reshape, then transpose conserves the ordering of "words" within the "sentence"
        ## attn is calculated independently for each Nth 64 elements of the embeddings of a sentence
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))] # 1st arg to zip are 4 nn.Linears, this only uses the first 3

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.

        ## contiguous() copies memory
        ## stiches weighted values back from [30, 8, 10, 64] to [30, 10, 512]
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x) # the last nn.Linears is used here, I guess.
