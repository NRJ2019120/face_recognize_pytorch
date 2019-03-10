from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms
from torch .utils import data
import numpy as np
import torch

def to_one_hot(lebals):
    lebals=lebals.numpy()
    # print(lebals)
    batch,i = lebals.shape
    # print(batch,i)
    out = np.zeros(shape=(100,100))
    for i in range(batch):
        index = int(lebals[i][0])
        # print(index)
        out[i][index]+=1
    return out
# if __name__ == '__main__':
#     tensor = torch.Tensor([[2],[1],[7],[9]])
#     out = to_one_hot(tensor)
#     print(out)
# lebals = torch.Tensor([0,5,7]).long()
# print(lebals)
# output =  torch.randn(1,10)
# print(output,output.shape)
# out=torch.cat((output,output),dim=0)
# print(out,out.shape)
# print(output[lebals])
# input = torch.Tensor(np.arange(5).reshape(1,5))
# target = input*10
# target = torch.cat(([input]*10),dim=0)
# # target = torch.Tensor([[np.arange(10)]*10.reshape(1,10)])
# print(input,input.shape)
# print(target,target.shape)

# x = np.reshape(np.array([np.arange(7)]*7), [ 7, 7])
# print(x,x.shape)

#
# loss_fn = torch.nn.MSELoss(reduce=False, size_average=False)
# # loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)
# #loss_fn = torch.nn.MSELoss()
# input = torch.randn(100,128)
# target = torch.randn(100,128)
# loss = loss_fn(input, target)
# # loss_sum = torch.sum(loss,dim=1)
# print(input); print(target); print(loss)
# print(input.size(), target.size(), loss.size())

#
# im = Image.open("/media/tensorflow01/myfile/100-WebFace/2/005.jpg")
# im.show()
# imgdata = np.array(im)
# print(imgdata)
# im_2 = Image.fromarray(imgdata)
# print(type(imgdata))
# print(imgdata.shape)
# im_2.show()

x = torch.randn(100,128)
print(x)