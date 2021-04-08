import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable

from model.position import PositionalEncoding, subsequent_mask
from util.greedy_decode import greedy_decode
from util.label_smoothing import LabelSmoothing
from util.noam_opt import NoamOpt
from util.run_epoch import run_epoch
from model.batch import Batch, batch_size_fn, rebatch, MyIterator
from device.CPU_loss import CPULossCompute
from device.multi_GPU_loss import MultiGPULossCompute
from model.transformer import make_model


def build_random_src_tgt(V, batch):
    data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
    data[:, 0] = 1  # why set first column to 1?
    src = Variable(data, requires_grad=False)
    tgt = Variable(data, requires_grad=False)

    # possibly Windows specific bugfix
    src = src.type(torch.LongTensor)
    tgt = tgt.type(torch.LongTensor)
    return src, tgt


def data_gen(V, batch, nbatches, listmode=False):
    "Generate random data for a src-tgt copy task."
    """
    V: data will be uniform random int from 1 to V, inclusive
    batch: size of a block of random data vectors (arrays batch x 10)
    nbatches: number of batches
    """
    for i in range(nbatches):
        src, tgt = build_random_src_tgt(V, batch)
        ## src and tgt are tensors with shape [batch, 10]
        ## in the copy task, batch = 30.
        yield Batch(src, tgt, 0)


def data_gen_list(V, batch, nbatches):
    data_batches = [0] * nbatches
    for i in range(nbatches):
        src, tgt = build_random_src_tgt(V, batch)
        data_batches[i] = Batch(src, tgt, 0)
    return data_batches


class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.data * norm


if __name__ == '__main__':

    misc_plots = False
    greedy_decoding_replicated = True
    # ERROR 1: (maybe windows/cpu only)
    # RuntimeError: Expected tensor for argument #1 'indices' to have scalar type Long; but got torch.IntTensor instead (while checking arguments for embedding)


    if misc_plots:
        plt.figure(figsize=(5,5))
        plt.imshow(subsequent_mask(20)[0])
        plt.show()

        plt.figure(figsize=(15, 5))
        pe = PositionalEncoding(20, 0)
        y = pe.forward(Variable(torch.zeros(1, 100, 20)))
        plt.plot(np.arange(100), y[0, :, 4:8].data.numpy())
        plt.legend(["dim %d" % p for p in [4,5,6,7]])
        plt.show()

        # Small example model.
        tmp_model = make_model(10, 10, 2)

        # Three settings of the lrate hyperparameters.
        opts = [NoamOpt(512, 1, 4000, None),
                NoamOpt(512, 1, 8000, None),
                NoamOpt(256, 1, 4000, None)]
        plt.plot(np.arange(1, 20000), [[opt.rate(i) for opt in opts] for i in range(1, 20000)])
        plt.legend(["512:4000", "512:8000", "256:4000"])
        plt.show()

        # Example of label smoothing.
        crit = LabelSmoothing(5, 0, 0.4)
        predict = torch.FloatTensor([[0, 0.2, 0.7, 0.1, 0],
                                     [0, 0.2, 0.7, 0.1, 0],
                                     [0, 0.2, 0.7, 0.1, 0]])
        v = crit(Variable(predict.log()),
                 Variable(torch.LongTensor([2, 1, 0])))

        # Show the target distributions expected by the system.
        plt.imshow(crit.true_dist)
        plt.show()

        crit = LabelSmoothing(5, 0, 0.1)
        def loss(x):
            d = x + 3 * 1
            predict = torch.FloatTensor([[0, x / d, 1 / d, 1 / d, 1 / d],
                                         ])
            #print(predict)
            return crit(Variable(predict.log()),
                         Variable(torch.LongTensor([1]))).data
        plt.plot(np.arange(1, 100), [loss(x) for x in range(1, 100)])
        plt.show()

    if greedy_decoding_replicated:

        # Train the simple copy task.
        epochs = 15
        V = 11  # input symbols are integers from 1 to 11 inclusive
        criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
        model = make_model(V, V, N=2)  # model is EncoderDecoder object
        model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
                            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

        # Generate random data
        data_batches = data_gen_list(V, 30, 5)  # old call: data_gen(V, 30, 20) (makes generator instead of list)
        print('data_batches properties')
        print(type(data_batches), 'lengths', len(data_batches), type(data_batches[0]), '\n')

        # Train the model
        loss_curve = [0] * epochs
        for epoch in range(epochs):  # def 10
            ## calls nn.Module.train() which sets mode to train
            model.train()
            epoch_loss = run_epoch(data_batches, model,
                                   SimpleLossCompute(model.generator, criterion, model_opt))
            print('Epoch %d loss:' % epoch, epoch_loss)
            loss_curve[epoch] = epoch_loss
            ## sets mode to testing (i.e. train=False).
            ## Layers like dropout behave differently depending on if mode is train or testing.
            model.eval()

        """print()
        tester1 = model.src_embed(src)
        print(type(tester1))
        tester2 = model.encoder(model.src_embed(src), src_mask)
        print(type(tester2))
        tester3 = model.encode(src, src_mask)
        print(type(tester3))"""

        model.to(torch.device('cpu'))
        model.eval()

        src = Variable(torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]))
        src_mask = Variable(torch.ones(1, 1, 10))
        print(greedy_decode(model, src, src_mask, max_len=10, start_symbol=1))
