#!E:\anaconda/python

import torch
from torch import nn

if __name__ == '__main__':
    m = nn.Softmax(dim=0)
    input = torch.randn(2, 3)
    print(input)

    output = m(input)
    print(output)
