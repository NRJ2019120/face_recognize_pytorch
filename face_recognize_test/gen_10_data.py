import os
import shutil

if __name__ == '__main__':

    casia_path = "/media/tensorflow01/myfile/CASIA-WebFace"
    train_path = "/media/tensorflow01/myfile/10-WebFace"
    list = os.listdir(casia_path)
    print(list)
    print(len(list))
    count =0
    for i in range(len(list)):
       # 挑选大于100 张的类别人脸  复制并重命名新文件
        file_list = os.listdir(os.path.join(casia_path, list[i]))
        if len(file_list)>100 and count<10:
            shutil.copytree(os.path.join(casia_path,list[i]),os.path.join(train_path,str(count)))
            count +=1
        if count==10:
            break