# from torchvision.models import resnet101
# import torch
# x = torch.ones([1,3,224,224])
# model = resnet101()
# out = model(x)
# print(model)
# print(out.shape)
# exit()
#
# from torchvision.models import resnet101
# from torchvision.models import vgg16
# x = torch.ones([1,3,224,224])
# model = vgg16()
# out = model(x)
# print(model)
# print(out.shape)
# exit()
#vgg缺点全连接层导致参数过多，
from torchvision.models import vgg16
import torch
x = torch.ones([1,3,224,224])
model = vgg16()
out = model(x)
print(model)
print(out.shape)
exit()
