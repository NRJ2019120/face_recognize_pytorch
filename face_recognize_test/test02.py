import torch
import torch.nn as nn

m = nn.Softmax(dim=1)
x = torch.randn(10,100)
output = m(x)
print(output)
print(output.shape)
print(torch.sum(output[1]))