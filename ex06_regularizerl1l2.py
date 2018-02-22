#coding=utf-8
''' Import library '''
# from __future__ import print_function
# from past.builtins import execfile
# execfile('00_readingInput.py')
import numpy as np
exec(open("00_readingInput.py").read())

''' Import l1,l2 (regularizer) '''
# (Do!) 從 keras.regularizer 中 import l1,l2 兩種 regularizer 

''' set the size of mini-batch and number of epochs'''
batch_size = 16
epochs = 50

''' Import keras to build a DL model '''
from keras.models import Sequential
from keras.layers.core import Dense, Activation

print('Building a model with regularizer L2')
model_l2 = Sequential()
# (Do!) 請加入 L2 regularizer
model_l2.add(Dense(128, input_dim=200))
model_l2.add(Activation('relu'))

# (Do!) 請加入 L2 regularizer
model_l2.add(Dense(256))
model_l2.add(Activation('relu'))

# (Do!) 請加入 L2 regularizer
model_l2.add(Dense(5))
model_l2.add(Activation('softmax'))

''' Setting optimizer as Adam '''
from keras.optimizers import SGD, Adam, RMSprop, Adagrad
model_l2.compile(loss= 'categorical_crossentropy',
              	optimizer='Adam',
              	metrics=['accuracy'])

'''Fit models and use validation_split=0.1 '''
history_l2 = model_l2.fit(X_train, Y_train,
							batch_size=batch_size,
							epochs=epochs,
							verbose=0,
							shuffle=True,
                    		validation_split=0.1)
loss_l2 = history_l2.history.get('loss')
acc_l2 = history_l2.history.get('acc')
val_loss_l2 = history_l2.history.get('val_loss')
val_acc_l2 = history_l2.history.get('val_acc')

# reference
print('Building a model without regularizer L2')
model_adam = Sequential()
model_adam.add(Dense(128, input_dim=200))
model_adam.add(Activation('relu'))
model_adam.add(Dense(256))
model_adam.add(Activation('relu'))
model_adam.add(Dense(5))
model_adam.add(Activation('softmax'))

''' Setting optimizer as Adam '''
from keras.optimizers import SGD, Adam, RMSprop, Adagrad
model_adam.compile(loss= 'categorical_crossentropy',
              		optimizer='Adam',
              		metrics=['accuracy'])

'''Fit models and use validation_split=0.1 '''
history_adam = model_adam.fit(X_train, Y_train,
							batch_size=batch_size,
							epochs=epochs,
							verbose=0,
							shuffle=True,
                    		validation_split=0.1)

''' Access the performance on validation data '''
loss_adam = history_adam.history.get('loss')
acc_adam = history_adam.history.get('acc')
val_loss_adam = history_adam.history.get('val_loss')
val_acc_adam = history_adam.history.get('val_acc')

''' Visualize the loss and accuracy of both models'''
''' Visualize the loss and accuracy of both models'''
import matplotlib.pyplot as plt
plt.figure(0,figsize=(8,6))
plt.subplot(121)
plt.plot(range(len(loss_adam)), loss_adam,label='Training')
plt.plot(range(len(val_loss_adam)), val_loss_adam,label='Validation')
plt.ylim([0,2])
plt.title('Loss - Original')
plt.xlabel("epoch")
plt.legend(loc='upper left')
plt.subplot(122)
plt.plot(range(len(loss_l2)), loss_l2,label='Training')
plt.plot(range(len(val_loss_l2)), val_loss_l2,label='Validation')
plt.ylim([0,2])
plt.title('Loss - With Regularizer')
plt.xlabel("epoch")
plt.tight_layout()
plt.savefig('06_regularizer_loss.png',dpi=300,format='png')
plt.close()
print('Result saved into 06_regularizer.png')

import matplotlib.pyplot as plt
plt.figure(0,figsize=(8,6))
plt.subplot(121)
plt.plot(range(len(acc_adam)), acc_adam,label='Training')
plt.plot(range(len(val_acc_adam)), val_acc_adam,label='Validation')
plt.ylim([0.5,0.95])
plt.title('Accuracy - Original')
plt.xlabel("epoch")
plt.legend(loc='upper left')
plt.subplot(122)
plt.plot(range(len(acc_l2)), acc_l2,label='Training')
plt.plot(range(len(val_acc_l2)), val_acc_l2,label='Validation')
plt.ylim([0.5,0.95])
plt.title('Accuracy - With Regularizer')
plt.xlabel("epoch")
plt.tight_layout()
plt.savefig('06_regularizer_acc.png',dpi=300,format='png')
plt.close()
print('Result saved into 06_regularizer.png')