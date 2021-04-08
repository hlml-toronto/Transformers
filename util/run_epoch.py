import time

def run_epoch(data_iter, model, loss_compute):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    ## data_iter is generator of Batch objects
    for i, batch in enumerate(data_iter):
        #print(len(list(data_iter)))  # <- this broke the loop by modifying data_iter possibly (got ~6051, then 40)
        ## this calls the forward() method of the EncoderDecoder class
        out = model.forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += float(batch.ntokens)
        tokens += float(batch.ntokens)
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                    (i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens
