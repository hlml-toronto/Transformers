import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable

from classes import make_model, subsequent_mask, NoamOpt, LabelSmoothing, PositionalEncoding, SimpleLossCompute, \
    run_epoch, data_gen

if __name__ == '__main__':

    misc_plots = False
    greedy_decoding_replicated = True

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

        # greedy decoding
        # Testing on Windows & CPU only
        # ISSUE TO FIX #1:
        # run_epoch() call -> RuntimeError: Expected tensor for argument #1 'indices' to have scalar type Long; but got torch.IntTensor instead (while checking arguments for embedding)

        # Train the simple copy task.
        V = 11  # input symbols are integers from 1 to 11 inclusive
        criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
        model = make_model(V, V, N=2)  # model is EncoderDecoder object
        model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
                            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

        for epoch in range(10):
            ## calls nn.Module.train() which sets mode to train
            model.train()
            run_epoch(data_gen(V, 30, 20), model,
                      SimpleLossCompute(model.generator, criterion, model_opt))
            ## sets mode to testing (i.e. train=False).
            ## Layers like dropout behave differently depending on if mode is train or testing.
            model.eval()
            print(run_epoch(data_gen(V, 30, 5), model,
                            SimpleLossCompute(model.generator, criterion, None)))

        # Train the simple copy task.
        """
        V = 11  # input symbols are integers from 1 to 11 inclusive
        criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
        model = make_model(V, V, N=2)  # model is EncoderDecoder object
        model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
                            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

        print()
        tester1 = model.src_embed(src)
        print(type(tester1))
        tester2 = model.encoder(model.src_embed(src), src_mask)
        print(type(tester2))
        tester3 = model.encode(src, src_mask)
        print(type(tester3))

        data_batches = data_gen_alt(V, 30, 20)
        for i, batch in enumerate(data_batches):
            print('%d...' % i)
            outTest = model.forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
            print(type(outTest))
        """


        def greedy_decode(model, src, src_mask, max_len, start_symbol):
            memory = model.encode(src, src_mask)
            ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
            for i in range(max_len - 1):
                out = model.decode(memory, src_mask,
                                   Variable(ys),
                                   Variable(subsequent_mask(ys.size(1))
                                            .type_as(src.data)))
                prob = model.generator(out[:, -1])
                _, next_word = torch.max(prob, dim=1)
                next_word = next_word.data[0]
                ys = torch.cat([ys,
                                torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
            return ys


        model.to(torch.device('cpu'))
        model.eval()

        src = Variable(torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]))
        src_mask = Variable(torch.ones(1, 1, 10))
        print(greedy_decode(model, src, src_mask, max_len=10, start_symbol=1))
