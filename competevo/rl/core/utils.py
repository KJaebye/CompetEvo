

def init_fc_weights(fc):
    fc.weight.data.mul_(0.1)
    fc.bias.data.mul_(0.0)