import torch
from dataset import FaceDataset
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import torch.utils.data as data
import os,shutil

def to_one_hot(lebals):

    lebals = lebals.numpy()
    # print(lebals)           #一维【】
    batch = lebals.size     #batchsize = 100
    # print(batch)
    out = np.zeros(shape=(BATCH_SIZE,10))
    for i in range(batch):
        index = int(lebals[i])
        # print(index)
        out[i][index] = 1
    out = torch.Tensor(out)
    return out
# lebals = torch.Tensor([1,2,3,4,5,60,7,8,99,10])
# print(to_one_hot(lebals).shape)
# exit()
def init_weights(m):                #参数初始化
    if isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear):
        nn.init.xavier_uniform(m.weight)  #-1~1
        nn.init.constant(m.bias,0.1)

class NET(nn.Module):
    def __init__(self):
        super(NET, self).__init__()
        self.center = torch.nn.Parameter(torch.randn(10,64)) #随机定义人脸类别中心点
        self.soft_fun = torch.nn.Softmax(dim=1)
        # print(self.center)
        self.layer1 = nn.Sequential(
            # 1-1 conv layer
            nn.Conv2d(3, 64, kernel_size=3, padding=1),       #输入是(3,,224,224）
            nn.BatchNorm2d(64),
            nn.PReLU(),

            # 1-2 conv layer
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),

            # 1 Pooling layer
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(

            # 2-1 conv layer
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),

            # 2-2 conv layer
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),

            # 2 Pooling lyaer
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(

            # 3-1 conv layer
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),

            # 3-2 conv layer
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),

            # 3 Pooling layer
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer4 = nn.Sequential(

            # 4-1 conv layer
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.PReLU(),

            # 4-2 conv layer
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.PReLU(),

            # 4 Pooling layer
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer5 = nn.Sequential(

            # 5-1 conv layer
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.PReLU(),

            # 5-2 conv layer
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.PReLU(),

            ## 5-3 conv layer
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.PReLU(),

            # 5 Pooling layer
            nn.MaxPool2d(kernel_size=2, stride=2))   # features 7*7*512

        self.layer6 = nn.Sequential(

            # 6 Fully connected layer
            # Dropout layer omitted since batch normalization is used.
            nn.Linear(7*7*512, 1024),
            nn.BatchNorm1d(1024),
            nn.PReLU(),
            nn.Dropout(0.5))

        self.layer7 = nn.Sequential(

            # 7 Fully connected layer
            # Dropout layer omitted since batch normalization is used.
            nn.Linear(1024, 64))
            # nn.BatchNorm1d(128))  #特征向量不需要 batchnormal

        self.layer8 = nn.Sequential(
            # 8 output layer
            nn.PReLU(),
            nn.Linear(64, 10))   #损失函数有softmax自动归一化效果

    def forward(self,x):               #输入【N，C，H，W】

        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)            #7*7*512
        out = out.view(out.size(0), -1)   #全连接 维度拉长
        out = self.layer6(out)
        center_out = self.layer7(out)      #128
        outputs = self.layer8(center_out)  #10
        # outputs = self.soft_fun(outputs)   #使用交叉熵时，已经包含softmax()
        # print(outputs.shape,"+==========")
        return center_out,outputs

if __name__ == '__main__':
    BATCH_SIZE = 8
    paramers_path ="/home/tensorflow01/oneday/face_recognize_test/facenet_Cross.pkl" #两种损失函数比较
    # paramers_path ="/home/tensorflow01/oneday/face_recognize_test/facenet.pkl"
    train_path = "/media/tensorflow01/myfile/10-WebFace"
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  #图像随机翻转
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225]),
    ])
    dataset = FaceDataset(train_path)
    trainLoader = data.DataLoader(dataset=dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=4,drop_last=True)
    facenet = NET()
    facenet.train()
    facenet.cuda()
    # Loss and Optimizer
    # classify_fun = nn.MSELoss(size_average=False,reduce=True)     #两种损失函数比较
    classify_fun = nn.CrossEntropyLoss(size_average=True,reduce=True)

    center_fun = nn.MSELoss(size_average=True, reduce=True)
    optimizer = torch.optim.Adam(facenet.parameters(),lr=0.0005,weight_decay=0.001)   #学习率超过0.01时，报nan

    if os.path.exists(paramers_path):
        facenet.load_state_dict(torch.load(paramers_path))
        print("model restore")
    if not os.path.exists(paramers_path):
        dirs, file = os.path.split(paramers_path)
        if not os.path.exists(dirs):
            os.makedirs(dirs)
    # Train the model
    for _ in range(10):
        for epoch,(imgdata,lebals) in enumerate(trainLoader):
            print(epoch)
            # print(lebals.shape)
            lebals = lebals.cuda()
            # lebals_hot = to_one_hot(lebals).cuda() #MSE需要转one_hot,nn.CrossEntropyLoss(),则不需要
            # print(lebals_hot.shape)
            # print(lebals,lebals.type())
            # exit()
            imgdata = imgdata.cuda()
            center_out,outputs = facenet(imgdata)
            # print(outputs.shape,lebals_hot.shape)
            # exit()
            # classify_loss = classify_fun(outputs,lebals_hot)   #均方差
            classify_loss = classify_fun(outputs,lebals)       #交叉熵

            # Expected object of type torch.cuda.LongTensor but found type torch.cuda.FloatTensor for argument #2 'target'
            center_loss = (1/BATCH_SIZE)*center_fun(center_out,facenet.center[lebals]) #距离量度方式有不同 如欧式距离，余弦相似度
            # print(center_out.shape ,facenet.center[lebals].shape)
            # exit()
            total_loss = classify_loss + 0.2*center_loss            #超参数调整与设置很重要
            print("--total_loss==>", total_loss.item(),"--classify_loss==>", classify_loss.item(),"--center_loss==>", center_loss.item())
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if (epoch+1)%100 == 0:    #验证正确率
                torch.save(facenet.state_dict(), paramers_path)
                print("module save")
                # 复制参数以防参数文件训练过程中损坏
                shutil.copyfile(paramers_path, os.path.join(r"/home/tensorflow01/oneday/face_recognize_test",
                                                                    "copy_facenet_Cross.pkl"))  # DST必须是完整的目标文件名
                # shutil.copyfile(paramers_path, os.path.join(r"/home/tensorflow01/oneday/face_recognize_test",
                #                                          "copy_facenet.pkl"))  # DST必须是完整的目标文件名

                print("--total_loss==>", total_loss.item(), "--classify_loss==>", classify_loss.item(), "--center_loss==>",center_loss.item())
                index1 = torch.argmax(outputs, dim=1)     # 注意此处维度错误
                # index2 = torch.argmax(lebals_hot,dim=1)
                index2 = lebals.cuda()
                # print(outputs.shape,lebals.shape)
                # print(outputs)
                print(index1, index2)
                sum = index1.eq(index2).cpu().sum()
                print(sum)
                print("features_out", center_out.shape)
                print("net_center", facenet.center[lebals].shape)
                print("facenet.center",facenet.center)
                accuracy = sum.float() / BATCH_SIZE
                print("accuracy==>", accuracy)
                print("***********************")
                # break
