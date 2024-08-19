import h5py
import pandas as pd
import numpy as np
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU, Dropout, Bidirectional
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import keras.backend as K
from keras.callbacks import LearningRateScheduler
from sklearn.decomposition import PCA
import joblib
import time

# ###Data Reading###
# Data paths
Xtrainpath = './data/Xtrain.mat'
Ytrainpath = './data/Ytrain.mat'
model_path = './result/model.h5'
# Reading data
xdata = h5py.File(Xtrainpath, mode='r')['Xtrain']
ydata = h5py.File(Ytrainpath, mode='r')['Ytrain']

# Transpose and convert data to array
xdata = np.transpose(xdata)
ydata = np.transpose(ydata)

# Perform PCA dimensionality reduction on data and save transformation matrix
pca1 = PCA(n_components=12)
Xtrain = pca1.fit_transform(xdata) # Xtrain is the reduced training set
joblib.dump(pca1, 'pca.m') # Save the model to pca.m file

# ###Data Normalization###
scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(Xtrain)
joblib.dump(scaler, 'scaler.m') # Save the model to scaler.m file
Xtrain = scaler.transform(Xtrain)
print(Xtrain)
print(ydata)

# Reshape input data for LSTM recognition
Xtrain = Xtrain.reshape(Xtrain.shape[0], Xtrain.shape[1], 1)
# print("Xtrain:", Xtrain.shape, "Ytrain:", Ytrain.shape)

in_dim = (Xtrain.shape[1], Xtrain.shape[2])
out_dim = ydata.shape[1]
print(in_dim)
print(out_dim)

model = Sequential()
model.add(LSTM(units=64,  return_sequences=True, input_shape=in_dim))
model.add(LSTM(units=32,  return_sequences=True, input_shape=in_dim))
model.add(LSTM(units=16, input_shape=in_dim))
model.add(Dense(out_dim))

# Define optimizer and parameters, sgd: lr for learning rate, decay for decay rate, momentum for momentum parameter (greater than or equal to zero)
opt = optimizers.adam_v2.Adam(learning_rate=0.005, beta_1=0.9, beta_2=0.999, amsgrad=False)

def scheduler(epoch):
    # Every 100 epochs, reduce the learning rate to 0.8 times the original
    if epoch % 500 == 0 and epoch != 0:
        learning_rate = K.get_value(model.optimizer.learning_rate)
        K.set_value(model.optimizer.learning_rate, learning_rate * 0.8)
        print("lr changed to {}".format(learning_rate * 0.8))
    return K.get_value(model.optimizer.lr)

reduce_lr = LearningRateScheduler(scheduler)

# Model compilation
model.compile(loss="mse", optimizer=opt, metrics=["mae"])
model.summary()

# Termination conditions for training, need to be included in callbacks
monitor_val_acc = EarlyStopping(monitor='val_mae', min_delta=0, patience=1000, mode='min', verbose=1, restore_best_weights = True)

# Save the model using checkpoints
filepath = "./result/weights-improvement-{epoch:02d}-{val_mae:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='mae', verbose=1, save_best_only=True, mode='min')
callbacks_list = [reduce_lr]

# ###Model Training###
start = time.time()
history = model.fit(Xtrain, ydata, epochs=22289, batch_size=40000,
          verbose=1,
          validation_split=0.2,
          shuffle=True,
          callbacks=callbacks_list)
end = time.time()
runTime = end - start
print(runTime)
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist['epoch'] = hist['epoch']+1
hist.to_csv('./result/result.csv', mode='a', header=False, index=False)

# ###Save the Model###
model.save(model_path)

# Loss curve
fig = plt.figure(figsize=(10, 10))
fig.add_subplot(1, 2, 1)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='test_loss')
plt.title('Model loss')
plt.ylabel('Value')
plt.xlabel('Epoch')
plt.title('Training and Validation Loss')