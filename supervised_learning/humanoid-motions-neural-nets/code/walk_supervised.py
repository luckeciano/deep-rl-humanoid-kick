import numpy as np
import pandas as pd

from keras.layers import Input, Dense, LeakyReLU
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras import backend as K
from keras.utils.vis_utils import plot_model

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import mean_absolute_error

from tensorflow.python.client import device_lib

#Dataset processing
dataset = pd.read_csv('ut_walk2.txt', delimiter = "::", engine = 'python')
	
train_X = dataset.values[0:, 0] - dataset.values[0,0] #Normalize the time
train_Y = dataset.values[0:, 1:23]
train_Y[0:, 0] = 0
train_Y[0:, 1] = 0
#As we just want to get periods from the dataset, we change the time instants. Check the dataset
for i in range(train_X.shape[0]):
	train_X[i] = 0.02*(i%8) 


has_ended = np.zeros((train_X.shape[0],1))
has_ended[train_X.shape[0] - 1] = 1
train_Y = np.concatenate((train_Y, has_ended), axis=1)

#Model Design
model =  Sequential()
model.add(Dense(75, input_dim = 1))
model.add(LeakyReLU())
model.add(Dense(50))
model.add(LeakyReLU())
model.add(Dense(23))
model.summary()
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=False)

#Training Procedure
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay = 0.0)
model.compile (loss = 'mean_squared_error', optimizer = adam, metrics = ['mse', 'mae'])
history_callback = model.fit (train_X, train_Y, epochs = 10000, batch_size = 128, verbose = 2)#, validation_data = (test_X, test_Y))
loss_hist_mse = np.array(history_callback.history["mean_squared_error"])
loss_hist_mae = np.array(history_callback.history["mean_absolute_error"])

adam = Adam(lr=0.0008, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile (loss = 'mean_squared_error', optimizer = adam, metrics = ['mse', 'mae'])
history_callback = model.fit (train_X, train_Y, epochs = 1000, batch_size = 128, verbose = 2)#, validation_data = (test_X, test_Y))
loss_hist_mse = np.concatenate((loss_hist_mse, np.array(history_callback.history["mean_squared_error"])))
loss_hist_mae = np.concatenate((loss_hist_mae, np.array(history_callback.history["mean_absolute_error"])))


adam = Adam(lr=0.0006, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile (loss = 'mean_squared_error', optimizer = adam, metrics = ['mse', 'mae'])
history_callback = model.fit (train_X, train_Y, epochs = 1000, batch_size = 128, verbose = 2)#, validation_data = (test_X, test_Y))
loss_hist_mse = np.concatenate((loss_hist_mse, np.array(history_callback.history["mean_squared_error"])))
loss_hist_mae = np.concatenate((loss_hist_mae, np.array(history_callback.history["mean_absolute_error"])))


adam = Adam(lr=0.0004, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile (loss = 'mean_squared_error', optimizer = adam, metrics = ['mse', 'mae'])
history_callback = model.fit (train_X, train_Y, epochs = 1000, batch_size = 128, verbose = 2)#, validation_data = (test_X, test_Y))
loss_hist_mse = np.concatenate((loss_hist_mse, np.array(history_callback.history["mean_squared_error"])))
loss_hist_mae = np.concatenate((loss_hist_mae, np.array(history_callback.history["mean_absolute_error"])))


adam = Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile (loss = 'mean_squared_error', optimizer = adam, metrics = ['mse', 'mae'])
history_callback = model.fit (train_X, train_Y, epochs = 1000, batch_size = 128, verbose = 2)#, validation_data = (test_X, test_Y))
loss_hist_mse = np.concatenate((loss_hist_mse, np.array(history_callback.history["mean_squared_error"])))
loss_hist_mae = np.concatenate((loss_hist_mae, np.array(history_callback.history["mean_absolute_error"])))

np.savetxt("mae_history_kick_policy.txt", loss_hist_mae, delimiter='\n')
np.savetxt("mse_history_kick_policy.txt", loss_hist_mse, delimiter='\n')

print (K.get_session().graph)
model.save('neural_walk')
