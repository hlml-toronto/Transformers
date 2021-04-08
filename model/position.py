import torch
import numpy as np
import math

from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model) # size: [5000, 512]
        position = torch.arange(0, max_len).unsqueeze(1) # size: [5000, 1]
        ## not sure why this is implemented "in log space". Otherwise matches paper
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model)) # exp{2i*-ln(10000)/512}
        pe[:, 0::2] = torch.sin(position * div_term) # even columns set to sine
        pe[:, 1::2] = torch.cos(position * div_term) # odd columns set to cosine
        pe = pe.unsqueeze(0) # shape: [1, 5000, 512]

        ## registers pe as a buffer that should not to be considered a model parameter.
        ## Buffers, by default, are persistent and will be saved alongside parameters.
        ## Often used for running averages
        self.register_buffer('pe', pe)

    def forward(self, x): ## takes normalized, embedded x

        # x.shape is [30, 10, 512] or [30, 9, 512]
        # pe added is [1, 10, 512] or [1, 9, 512]
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x) # applies dropout (zeros some elements of x with prob=dropout)
