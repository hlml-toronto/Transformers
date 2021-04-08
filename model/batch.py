import numpy as np
import torch
from torch.autograd import Variable
from torchtext.legacy import data, datasets

from model.position import subsequent_mask


class Batch:
    """Object for holding a batch of data with mask during training."""

    def __init__(self, src, trg=None, pad=0):
        self.src = src # shape: [30, 10]
        self.src_mask = (src != pad).unsqueeze(-2) # masks padding, shape: [30, 1, 10]

        if trg is not None:
            self.trg = trg[:, :-1]  # cuts out last column of trg (why?)
            self.trg_y = trg[:, 1:] # cuts out first colum of trg (all 1s)
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        """Create a mask to hide padding and future words."""
        tgt_mask = (tgt != pad).unsqueeze(-2)
        ## tgt has shape [30, 9], after unsqueeze has shape [30, 1, 9]
        ## tgt_mask is true wherever tgt is not equal to pad
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        # type_as sets type of return value of subsequent_mask to same as tgt_mask.data
        # & takes intersection of two sets, final shape is [30, 9, 9]
        return tgt_mask


class MyIterator(data.Iterator):
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b

            self.batches = pool(self.data(), self.random_shuffler)

        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size, self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))


global max_src_in_batch, max_tgt_in_batch
def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


def data_gen(V, batch, nbatches):
    # "Generate random data for a src-tgt copy task."
    # generates nbatches of Batch objects
    for i in range(nbatches):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
        data[:, 0] = 1 # why is first column set to 1?
        # src = Variable(data, requires_grad=False)
        # tgt = Variable(data, requires_grad=False)
        src = data
        tgt = data
        ## src and tgt are tensors with shape [batch, 10]
        ## in the copy task, batch = 30.
        yield Batch(src, tgt, 0)


def rebatch(pad_idx, batch):
    "Fix order in torchtext to match ours"
    src, trg = batch.src.transpose(0, 1), batch.trg.transpose(0, 1)
    return Batch(src, trg, pad_idx)
