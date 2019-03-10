import torch
import torchvision.transforms as transforms
import os
from VGG16net import NET
import PIL.Image as Image

transform = transforms.Compose([
            transforms.Resize([224,224]),  # resize(224,224)
            transforms.RandomHorizontalFlip(),  # 图像随机翻转
            transforms.ToTensor()])
model = NET()
model.eval()
# paramers_path = "/home/tensorflow01/oneday/face_recognize_test/facenet.pkl"
paramers_path = "/home/tensorflow01/oneday/face_recognize_test/facenet_Cross.pkl"
model.load_state_dict(torch.load(paramers_path))
print("model restore")
count = 0
right = 0
distance_thresh = 1

for i in range(10,20):
    # file_path = r"/media/tensorflow01/myfile/10-WebFace/test_{}".format(i) #系统人脸新人脸 测试数据集
    file_path = r"/media/tensorflow01/myfile/10-WebFace/{}".format(i)         #系统外人脸 10-35 未训练的类别人脸
    images = os.listdir(file_path)
    for img in images[:10]:
        test_path = os.path.join(file_path,img)
        img = Image.open(test_path)
        img = img.convert('RGB')
        img_data = transform(img)
        # print(img_data.shape)
        img_data = img_data.unsqueeze(0)
        # print(img_data.shape)
        center_out,output = model(img_data)
        count +=1
        distance = torch.sum((model.center - center_out)**2,dim=1)
        # print(model.center)
        # print(center_out.shape)
        # print(model.center.shape)
        # print(distance.shape)
        print(distance)
        index1 = torch.argmin(distance).item()
        min_distance = torch.min(distance)     #而且滿足<阈值（超参数）
        thresh = 1
        # print(min_distance)
        index2 = torch.argmax(output).item()
        index3 = i
        print(index1, index2,index3)
        # if index1 == index3 and min_distance < 1:
        #     right +=1
        #     print("I recognize you",index1,index3,min_distance)  #对熟人，经过训练的人脸识别率100%
        # print(min_distance)                                     #
        if min_distance > 1:                                       #陌生人脸检测,最近距离大于同类阈值视为陌生人脸  80%
            right +=1
            print("I have not see you",min_distance)
print("count=",count)
print("accuracy=",right/count)                          #