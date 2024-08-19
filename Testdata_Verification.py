import numpy as np
import h5py
import scipy.io
import matplotlib.pyplot as plt
from keras.models import load_model
import joblib

# ###读取数据###
# 数据路径
Xtestpath = './data/OS.mat'
Ytestpath = './data/Wa.mat'
model_path = './result/model.h5'
# 读取
model = load_model(model_path)
Xtest = h5py.File(Xtestpath, mode='r')['OS']
Ytest = h5py.File(Ytestpath, mode='r')['Wa']

# 将数据进行转置并转为array
Xtest = np.transpose(Xtest)
Ytest = np.transpose(Ytest)

# 将测试数据利用Xtrain的pca进行降维
pca2 = joblib.load('pca.m') # 读入模型pca.m
Xtest = pca2.transform(Xtest)

# ###对数据进行归一化###
scaler = joblib.load('scaler.m') # 读入模型pca.m
Xtest = scaler.transform(Xtest)
# 将数据转换为LSTM网络可识别的结构
Xtest = Xtest.reshape(Xtest.shape[0], Xtest.shape[1], 1)

# ###通过模型进行预测###
ypred = model.predict(Xtest)
# ###模型预测结果与真实结果比较并绘图
x_ax = range(len(Xtest))
plt.title("Bi-LSTM multi-output prediction")
plt.scatter(x_ax, (Ytest[:, 0]-ypred[:,  0])*1000,  s=6, label="y1-test")
plt.scatter(x_ax, (Ytest[:, 1]-ypred[:, 1])*1000,  s=6, label="y2-test")
plt.legend()
plt.show()

scipy.io.savemat('ypred.mat', {'ypred': ypred})