import torch
import torch.nn as nn
from torch.autograd import Variable
from torchtext import data, datasets

from classes import Batch, run_epoch, batch_size_fn, LabelSmoothing, greedy_decode, NoamOpt, make_model


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
            for b in data.batch(self.data(), self.batch_size,
                                self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))


def rebatch(pad_idx, batch):
    "Fix order in torchtext to match ours"
    src, trg = batch.src.transpose(0, 1), batch.trg.transpose(0, 1)
    return Batch(src, trg, pad_idx)


# Skip if not interested in multigpu.
class MultiGPULossCompute:
    "A multi-gpu loss compute and train function."

    def __init__(self, generator, criterion, devices, opt=None, chunk_size=5):
        # Send out to different gpus.
        self.generator = generator
        self.criterion = nn.parallel.replicate(criterion,
                                               devices=devices)
        self.opt = opt
        self.devices = devices
        self.chunk_size = chunk_size

    def __call__(self, out, targets, normalize):
        total = 0.0
        generator = nn.parallel.replicate(self.generator,
                                          devices=self.devices)
        out_scatter = nn.parallel.scatter(out,
                                          target_gpus=self.devices)
        out_grad = [[] for _ in out_scatter]
        targets = nn.parallel.scatter(targets,
                                      target_gpus=self.devices)

        # Divide generating into chunks.
        chunk_size = self.chunk_size
        for i in range(0, out_scatter[0].size(1), chunk_size):
            # Predict distributions
            out_column = [[Variable(o[:, i:i + chunk_size].data,
                                    requires_grad=self.opt is not None)]
                          for o in out_scatter]
            gen = nn.parallel.parallel_apply(generator, out_column)

            # Compute loss.
            y = [(g.contiguous().view(-1, g.size(-1)),
                  t[:, i:i + chunk_size].contiguous().view(-1))
                 for g, t in zip(gen, targets)]
            # print("y object in multiGPU", type(y), y)
            loss = nn.parallel.parallel_apply(self.criterion, y)
            # print("loss object in multiGPU", type(loss), loss)

            # Sum and normalize loss
            l = nn.parallel.gather(loss,
                                   target_device=self.devices[0])
            # print("l object in multiGPU", type(l), l)
            # print("l.sum() object in multiGPU", type(l.sum()), l.sum())
            # print("l.data object in multiGPU", type(l.data), l.data)
            l = l.sum() / normalize  # switched numerator from l.sum()[0]
            total += l.data.item()  # switched from l.data[0]

            # Backprop loss to output of transformer
            if self.opt is not None:
                l.backward()
                for j, l in enumerate(loss):
                    out_grad[j].append(out_column[j][0].grad.data.clone())

        # Backprop all loss through transformer.
        if self.opt is not None:
            out_grad = [Variable(torch.cat(og, dim=1)) for og in out_grad]
            o1 = out
            o2 = nn.parallel.gather(out_grad,
                                    target_device=self.devices[0])
            o1.backward(gradient=o2)
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return total * normalize


if __name__ == '__main__':

    if True:

        import spacy
        spacy_de = spacy.load('de')
        spacy_en = spacy.load('en')

        def tokenize_de(text):
            return [tok.text for tok in spacy_de.tokenizer(text)]

        def tokenize_en(text):
            return [tok.text for tok in spacy_en.tokenizer(text)]

        BOS_WORD = '<s>'
        EOS_WORD = '</s>'
        BLANK_WORD = "<blank>"
        SRC = data.Field(tokenize=tokenize_de, pad_token=BLANK_WORD)
        TGT = data.Field(tokenize=tokenize_en, init_token = BOS_WORD,
                         eos_token = EOS_WORD, pad_token=BLANK_WORD)

        MAX_LEN = 100
        train, val, test = datasets.IWSLT.splits(
            exts=('.de', '.en'), fields=(SRC, TGT),
            filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and
                len(vars(x)['trg']) <= MAX_LEN)
        MIN_FREQ = 2
        SRC.build_vocab(train.src, min_freq=MIN_FREQ)
        TGT.build_vocab(train.trg, min_freq=MIN_FREQ)

    # MODIFIED: original values were
    # devices = [0, 1, 2, 3]
    # BATCH_SIZE = 12000

    # GPUs to use
    # devices = list(range(torch.cuda.device_count()))
    devices = [torch.device('cuda:%d' % a) for a in range(torch.cuda.device_count())]

    if torch.cuda.is_available():
        pad_idx = TGT.vocab.stoi["<blank>"]
        model = make_model(len(SRC.vocab), len(TGT.vocab), N=6)
        model.to(torch.device('cuda:0'))  # model.cuda()
        criterion = LabelSmoothing(size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1)
        criterion.cuda()
        BATCH_SIZE = 750  # 12000, note 12000 / 16 = 750
        train_iter = MyIterator(train, batch_size=BATCH_SIZE, device=torch.device('cuda:0'),  # changed device
                                repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                                batch_size_fn=batch_size_fn, train=True)
        valid_iter = MyIterator(val, batch_size=BATCH_SIZE, device=torch.device('cuda:0'),  # changed device
                                repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                                batch_size_fn=batch_size_fn, train=False)
        model_par = nn.DataParallel(model, device_ids=devices)

    # MODIFIED:
    #     model_opt = NoamOpt(model.src_embed[0].d_model, 1, 2000,
    #           torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    from GPUtil import showUtilization as gpu_usage
    print(model.src_embed[0].d_model)
    #torch.cuda.empty_cache()
    gpu_usage()

    TRAIN_MODEL = True
    NUM_EPOCHS = 1 # default 10
    if TRAIN_MODEL:
        model_opt = NoamOpt(model.src_embed[0].d_model, 1, 20,
                torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
        for epoch in range(NUM_EPOCHS):
            print('\nEpoch:', epoch, 'usage...')
            gpu_usage()
            model_par.train()
            run_epoch((rebatch(pad_idx, b) for b in train_iter),
                      model_par,
                      MultiGPULossCompute(model.generator, criterion,
                                          devices=devices, opt=model_opt))
            model_par.eval()
            loss = run_epoch((rebatch(pad_idx, b) for b in valid_iter),
                              model_par,
                              MultiGPULossCompute(model.generator, criterion,
                              devices=devices, opt=None))
            print(loss)
    else:
        model = torch.load("iwslt.pt")

    # USAGE CHECK
    # torch.cuda.empty_cache()
    gpu_usage()
    model.to(torch.device('cpu'))
    print()
    gpu_usage()

    # move model and inputs to gpu (currently 'cpu' fails here -- why?)
    device_to_assess_with = devices[0]  # 'cpu' or devices[0]
    model.to(device_to_assess_with)

    # full pass of input
    for i, batch in enumerate(valid_iter):
        src = batch.src.transpose(0, 1)[:1]
        src_mask = (src != SRC.vocab.stoi["<blank>"]).unsqueeze(-2)
        print(src.is_cuda)
        # out = greedy_decode(model, src, src_mask,
        #                    max_len=60, start_symbol=TGT.vocab.stoi["<s>"])
        out = greedy_decode(model, src, src_mask,
                            max_len=60, start_symbol=TGT.vocab.stoi["<s>"])

        print("Translation:", end="\t")
        for i in range(1, out.size(1)):
            sym = TGT.vocab.itos[out[0, i]]
            if sym == "</s>": break
            print(sym, end=" ")
        print()
        print("Target:", end="\t")
        for i in range(1, batch.trg.size(0)):
            sym = TGT.vocab.itos[batch.trg.data[i, 0]]
            if sym == "</s>": break
            print(sym, end=" ")
        print()
        break
