import os
import shutil

if __name__ == '__main__':

    train_path = "/media/tensorflow01/myfile/10-WebFace"
    list = os.listdir(train_path)
    # print(list)
    # print(len(list))
    for i in range(0,len(list)):
        # print(i)
        list2 = os.listdir(os.path.join(train_path,list[i]))
        file = open(os.path.join(train_path,list[i])+".txt", "w")
        for str2 in list2:
            # print(i+1)
            file.write(train_path+"/"+list[i]+"/"+ str2 +"  " + list[i]+"\n")
