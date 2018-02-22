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
from keras.layers.core import Dense, Activation

print('Building a model whose optimizer="Adam", activation function=relu')
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

loss_adam= history_adam.history.get('loss')
acc_adam = history_adam.history.get('acc')

print('Building a model whose optimizer="SGD", activation function=relu')
model_sgd = Sequential()
model_sgd.add(Dense(128, input_dim=200))
model_sgd.add(Activation('relu'))
model_sgd.add(Dense(256))
model_sgd.add(Activation('relu'))
model_sgd.add(Dense(5))
model_sgd.add(Activation('softmax'))

''' Setting optimizer as SGD '''
model_sgd.compile(loss='categorical_crossentropy',
				optimizer=SGD(lr=0.001),
				metrics=['accuracy'])

history_sgd = model_sgd.fit(X_train, Y_train,
							batch_size=batch_size,
							epochs=epochs,
							verbose=0,
							shuffle=True,
                    		validation_split=0.1)

loss_sgd= history_sgd.history.get('loss')
acc_sgd = history_sgd.history.get('acc')

''' Visualize the loss and accuracy of both models'''
import matplotlib.pyplot as plt
plt.figure(0,figsize=(8,6))
plt.subplot(121)
plt.plot(range(len(loss_adam)), loss_adam,label='Adam')
plt.plot(range(len(loss_sgd)), loss_sgd,label='SGD')
plt.title('Loss')
plt.xlabel("epoch")
plt.legend(loc='upper left')
plt.subplot(122)
plt.plot(range(len(acc_adam)), acc_adam,label='Adam')
plt.plot(range(len(acc_sgd)), acc_sgd,label='SGD')
plt.title('Accuracy')
plt.xlabel("epoch")
plt.tight_layout()
plt.savefig('04_optimizerSelection.png',dpi=300,format='png')
plt.close()
print('Result saved into 04_optimizerSelection.png')