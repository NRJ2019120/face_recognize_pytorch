"""

tensor([18.2648,  0.5033, 13.3199, 14.2500, 25.3589, 16.4621, 14.1144, 20.2351,14.5219, 11.9134]))
tensor([ 0.9893, 18.5803, 13.4524, 17.9496, 21.0007, 28.8765, 15.4412, 18.0086, 24.1712, 17.6958])

陌生人脸 = tensor([11.6734,  5.1733,  7.8566, 12.2868, 12.9758,  4.4882,  9.0119, 14.9156,10.2902,  3.0675])


 #对熟人，经过训练的人脸识别率98%
 #系统内测试 正确率 80%
#陌生人脸检测,最近距离大于同类阈值视为陌生人脸  80%
分析 ：阈值选择有待精确、特征提取器有待改进，样本粗糙，有噪声，超参数 y 还需要调整。。。
   距离量度方式的改进
"""