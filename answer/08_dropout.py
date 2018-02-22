#coding=utf-8
''' Import library '''
# from __future__ import print_function
# from past.builtins import execfile
# execfile('00_readingInput.py')
import numpy as np
exec(open("00_readingInput.py").read())

''' set the size of mini-batch and number of epochs'''
batch_size = 16
epochs = 100

''' Import keras to build a DL model '''
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation

print('Building current best model with Dropout = 0.4')
model_adam = Sequential()
model_adam.add(Dense(128, input_dim=200))
model_adam.add(Activation('relu'))
model_adam.add(Dropout(0.45))
model_adam.add(Dense(256))
model_adam.add(Activation('relu'))
model_adam.add(Dropout(0.45))
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

loss_adam= history_adam.history.get('loss')
acc_adam = history_adam.history.get('acc')
val_loss_adam = history_adam.history.get('val_loss')
val_acc_adam = history_adam.history.get('val_acc')

''' Visualize the loss and accuracy of both models'''
import matplotlib.pyplot as plt
plt.figure(0,figsize=(8,6))
plt.subplot(121)
plt.plot(range(len(loss_adam)), loss_adam,label='Training')
plt.plot(range(len(val_loss_adam)), val_loss_adam,label='Validation')
plt.title('Loss')
plt.xlabel("epoch")
plt.legend(loc='upper left')
plt.subplot(122)
plt.plot(range(len(acc_adam)), acc_adam,label='Training')
plt.plot(range(len(val_acc_adam)), val_acc_adam,label='Validation')
plt.title('Accuracy')
plt.xlabel("epoch")
plt.tight_layout()
plt.savefig('08_dropout.png',dpi=300,format='png')
plt.close()
print('Result saved into 08_dropout.png')