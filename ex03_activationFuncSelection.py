#coding=utf-8
''' Import library '''
# from __future__ import print_function
# from past.builtins import execfile
# execfile('00_readingInput.py')
import numpy as np
exec(open("00_readingInput.py").read())

''' set the size of mini-batch and number of epochs'''
batch_size = 16
epochs = 30

''' Import keras to build a DL model '''
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation

print('Building model using other activation functions')
''' Use other activation function '''
model_sp = Sequential()
model_sp.add(Dense(128, input_dim=200))
# (Do!) 加入 activation function
model_sp.add(Dense(256))
# (Do!) 加入 activation function
model_sp.add(Dense(5))
# (Do!) 加入 activation function

''' Use SGD(lr=0.01) as the optimizer  '''
''' lr set to 0.01 according to 02_learningRateSelection.py '''
from keras.optimizers import SGD
sgd = SGD(lr=0.01,momentum=0.0,decay=0.0,nesterov=False)

model_sp.compile(loss= 'categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

history_sp = model_sp.fit(X_train, Y_train,
							batch_size=batch_size,
							epochs=epochs,
							verbose=0,
							shuffle=True,
                    		validation_split=0.1)

loss_sp = history_sp.history.get('loss')
acc_sp = history_sp.history.get('acc')

# reference
model_bm = Sequential()
model_bm.add(Dense(128, input_dim=200))
model_bm.add(Activation('sigmoid'))
model_bm.add(Dense(256))
model_bm.add(Activation('sigmoid'))
model_bm.add(Dense(5))
model_bm.add(Activation('softmax'))
from keras.optimizers import SGD, Adam, RMSprop, Adagrad
sgd = SGD(lr=0.01,momentum=0.0,decay=0.0,nesterov=False)
model_bm.compile(loss='categorical_crossentropy',
				optimizer=sgd,
				metrics=['accuracy'])
history_bm = model_bm.fit(X_train, Y_train,
							batch_size=batch_size,
							epochs=epochs,
							verbose=0,
							shuffle=True,
                    		validation_split=0.1)
loss_bm	= history_bm.history.get('loss')
acc_bm 	= history_bm.history.get('acc')

import matplotlib.pyplot as plt
plt.figure(0,figsize=(8,6))
plt.subplot(121)
plt.plot(range(len(loss_sp)),loss_sp,label='relu')
plt.plot(range(len(loss_bm)),loss_bm,label='Sigmoid')
plt.title('Loss')
plt.xlabel("epoch")
plt.legend(loc='upper left')
plt.subplot(122)
plt.plot(range(len(acc_sp)),acc_sp,label='relu')
plt.plot(range(len(acc_bm)),acc_bm,label='Sigmoid')
plt.title('Accuracy')
plt.xlabel("epoch")
plt.tight_layout()
plt.savefig('03_activationFuncSelection.png',dpi=300,format='png')
plt.close()

print('Result saved into 03_activationFuncSelection.png')