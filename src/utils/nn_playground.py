#!E:\anaconda/python

import torch
from torch import nn
import os
from torchvision.io import write_video


if __name__ == '__main__':
    m = nn.Softmax(dim=0)
    input = torch.randn(2, 3)
    print(input)
    print(input.shape)

    output = torch.stack([input, input*2])
    print(output)

    output = m(input)
    print(output)

    mask = torch.rand(2, 2)
    print(mask)
    mask = torch.unsqueeze(mask, 0)
    print(mask)
    new_mask = torch.rand(2, 2, 2)
    print(new_mask)
    new_mask = torch.unsqueeze(new_mask, 1)
    print(new_mask)

    print(f"{input=}")
    if a := 0:
        print("OK")

    for idx, bb in enumerate(range(5)):
        print(idx, bb)

    c = ''
    print(c == '')

    d = "a.avi"
    dd = os.path.splitext(d)
    ddd = d.split('.')

    print(dd)
    print(ddd)

    r = torch.rand(1, 1, 300, 300)
    r *= 255
    o = torch.cat([r, r, r], dim=0)
    o = o.permute(0, 2, 3, 1)
    print(o.shape)
    o = torch.cat([o] * 3, dim=-1)
    print(o.shape)
    write_video(os.path.join(os.path.dirname(__file__), 'test.mp4'), o, fps=1)
    print("Save finished!")

    # create a random tensor of size (3, 3)
    # tensor = torch.rand(3, 3)

    r = torch.tensor([1, 2, 3])
    d = torch.tensor([2, 2, 2])
    print(r / d)

    c = torch.tensor([[1, 2, 3], [3, 2, 1]])
    print(c)
    print(c.sum(0))
    print(c.sum(1))

