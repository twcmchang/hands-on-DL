#coding=utf-8
''' Import library '''
# from __future__ import print_function
# from past.builtins import execfile
# execfile('00_readingInput.py')
import numpy as np
exec(open("00_readingInput.py").read())

''' Import keras to build a DL model '''
from keras.models import Sequential
from keras.layers.core import Dense, Activation

# print 'Building a model whose loss function is categorical_crossentropy'
''' For categorical_crossentropy '''
model = Sequential()
# (Do!) 加入 hidden layer of 128 neurons 與指定 input_dim=200
#       用 'sigmoid' 當作 activation function

# (Do!) 加入 hidden layer of 256 neurons
#		使用 'sigmoid' 當作 activation function

# (Do!) 加入 output layer of 5 neurons
# 		使用 'softmax'  當作 activation function


''' Set up the optimizer '''
from keras.optimizers import SGD, Adam, RMSprop, Adagrad
sgd = SGD(lr=0.01,momentum=0.0,decay=0.0,nesterov=False)

''' Compile model with specified loss and optimizer '''
# (Do!) 指定 loss function
model.compile(	loss='',
				optimizer=sgd,
				metrics=['accuracy'])

''' Set the size of mini-batch and number of epochs'''
''' Fit models and use validation_split=0.1 '''
# (Do!) 指定 batch_size, epochs, shuffle or not 
# 		與 validation_split 的比例
history = model.fit( X_train, # X_train
					 Y_train, # Y_train
					 batch_size=, # batch_size
					 epochs=, # epochs
					 shuffle=, # shuffle
					 validation_split=, # validation_split
					 verbose=0)	

'''Access the loss and accuracy in every epoch'''
loss	= history.history.get('loss')
acc 	= history.history.get('acc')

import matplotlib.pyplot as plt
plt.figure(0,figsize=(8,6))
plt.subplot(121)
plt.plot(range(len(loss_ce)), loss_ce,label='CE')
plt.title('Loss')
plt.legend(loc='upper left')
plt.subplot(122)
plt.plot(range(len(acc_ce)), acc_ce,label='CE')
plt.title('Accuracy')
plt.xlabel('epoch')
plt.savefig('00_firstModel.png',dpi=300,format='png')
plt.close()
print('Result saved into 00_firstModel.png')
