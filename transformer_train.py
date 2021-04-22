import spacy
import torch
import torch.nn as nn
from torchtext import data, datasets

from util.greedy_decode import greedy_decode
from util.label_smoothing import LabelSmoothing
from util.noam_opt import NoamOpt
from util.run_epoch import run_epoch
from model.batch import Batch, batch_size_fn, rebatch, MyIterator
from device.CPU_loss import CPULossCompute
from device.multi_GPU_loss import MultiGPULossCompute
from model.transformer import make_model

# TODO check

if __name__ == '__main__':

    print('In spacy loop...')
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
    TGT = data.Field(tokenize=tokenize_en, init_token=BOS_WORD,
                     eos_token=EOS_WORD, pad_token=BLANK_WORD)

    MAX_LEN = 100
    train, val, test = datasets.IWSLT.splits(
        exts=('.de', '.en'), fields=(SRC, TGT),
        filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and
                              len(vars(x)['trg']) <= MAX_LEN)
    MIN_FREQ = 2
    SRC.build_vocab(train.src, min_freq=MIN_FREQ)
    TGT.build_vocab(train.trg, min_freq=MIN_FREQ)
    print('Done spacy loop.')

    # MODIFIED: original values were
    # devices = [0, 1, 2, 3]
    # BATCH_SIZE = 12000

    # Devices / GPUs to use
    print('Check if torch.cuda.is_available():', torch.cuda.is_available())
    if torch.cuda.is_available():
        devices = [torch.device('cuda:%d' % a) for a in range(torch.cuda.device_count())]
        LossCompute = MultiGPULossCompute

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
    else:
        devices = [torch.device('cpu')]
        LossCompute = CPULossCompute

        pad_idx = TGT.vocab.stoi["<blank>"]
        model = make_model(len(SRC.vocab), len(TGT.vocab), N=6)
        model.to(torch.device('cpu'))
        criterion = LabelSmoothing(size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1)
        criterion.cpu()
        BATCH_SIZE = 750  # 12000, note 12000 / 16 = 750
        train_iter = MyIterator(train, batch_size=BATCH_SIZE, device=torch.device('cpu'),  # changed device
                                repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                                batch_size_fn=batch_size_fn, train=True)
        valid_iter = MyIterator(val, batch_size=BATCH_SIZE, device=torch.device('cpu'),  # changed device
                                repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                                batch_size_fn=batch_size_fn, train=False)
        model_par = nn.DataParallel(model, device_ids=devices)

    TRAIN_MODEL = True
    NUM_EPOCHS = 1  # default 10
    if TRAIN_MODEL:
        model_opt = NoamOpt(model.src_embed[0].d_model, 1, 20,
                torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
        for epoch in range(NUM_EPOCHS):
            print('\nEpoch:', epoch, 'usage...')
            model_par.train()
            run_epoch((rebatch(pad_idx, b) for b in train_iter),
                      model_par,
                      LossCompute(model.generator, criterion, devices=devices, opt=model_opt))
            model_par.eval()
            loss = run_epoch((rebatch(pad_idx, b) for b in valid_iter),
                              model_par,
                              LossCompute(model.generator, criterion, devices=devices, opt=None))
            print(loss)
    else:
        model = torch.load("iwslt.pt")

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
