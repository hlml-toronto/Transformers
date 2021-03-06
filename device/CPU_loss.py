class CPULossCompute:
    "A cpu loss compute and train function (copy contents of SimpleLossCompute)"

    def __init__(self, generator, criterion, devices, opt=None):
        assert len(devices) == 1
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
