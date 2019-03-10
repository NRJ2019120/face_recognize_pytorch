from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms
from torch .utils import data
import numpy as np

"创建数据集类"
class FaceDataset(Dataset):
    def __init__(self,path):
        self.transform = transforms.Compose([
            transforms.Resize([224,224]),  # resize(224,224)
            transforms.RandomHorizontalFlip(),  # 图像随机翻转
            transforms.ToTensor()])        #图像转成Tensor
        self.path = path
        self.dataset = []
        for i in range(0,10):   #测试十个人   100 个人训练不出分类，分类损失无法下降
            self.dataset.extend(open(os.path.join(self.path,"{0}.txt".format(i))).readlines())

    def __getitem__(self, index):
        strs = self.dataset[index].strip().split() #strip： 用来去除头尾字符、空白符(包括\n、\r、\t、' '，即：换行、回车、制表符、空格)
        # print(strs)
        img = Image.open(strs[0])
        img = img.convert('RGB')
        #RuntimeError: invalid argument 0: Sizes of tensors must match except in dimension 0. Got 3 and 1 in dimension 1
        # at /opt/conda/conda-bld/pytorch_1533672544752/work/aten/src/TH/generic/THTensorMath.cpp:3616
        img_data = self.transform(img)
        lebal = int(strs[1])    ##??????????????
        return img_data,lebal

    def __len__(self):
        return len(self.dataset)

if __name__ == '__main__':
    train_path = "/media/tensorflow01/myfile/10-WebFace"
    dataset = FaceDataset(train_path)
    # print(len(dataset))
    # print(dataset[0])  # 通道是3 imgdata ,con,offset
    # print(dataset[0][0].shape)  #torch.Size([3, 224, 224])
    # print(dataset)      #_<_main__.FaceDataset object at 0x00000000023C5CF8>
    dataloader = data.DataLoader(dataset, batch_size=10, shuffle=True,drop_last=True) #drop_last=True!!!!!
    while True:
        for epoch,(imgdata,lebal) in enumerate(dataloader):
            print(lebal.shape)
            print(imgdata.shape)
            print("epoch==>", epoch, "imgdata==>", imgdata, "lebal==>", lebal)
            imgdata=imgdata[0]*255
            imgdata = np.array(imgdata,dtype=np.uint8)
            print(imgdata)
            print(imgdata.shape)
            print(type(imgdata))
            imgdata= np.transpose(imgdata,(1,0,2))
            imgdata = np.transpose(imgdata,(0,2,1))
            print(imgdata.shape)
            im = Image.fromarray(imgdata)   #imgdata.shape()要从（3,224,224,）转成(224, 224, 3)！！！！！！！
            # im.show()
            break
        break
