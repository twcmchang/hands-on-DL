#coding=utf-8
''' Import library '''
# from __future__ import print_function
# from past.builtins import execfile
# execfile('00_readingInput.py')
import numpy as np
exec(open("00_readingInput.py").read())

''' EarlyStopping '''
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 3)

''' Self-defined Callbacks '''
from keras.callbacks import Callback
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []
    def on_epoch_end(self, batch, logs={}):
        self.loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))

loss_history=LossHistory()

''' Useful Callback: ModelCheckpoint '''
from keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint(filepath='model.h5',
                            monitor='val_loss',
                            mode='auto',
                            # period = 10,
                            save_best_only=True)

''' set the size of mini-batch and number of epochs'''
batch_size = 16
epochs = 50

''' Import keras to build a DL model '''
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation

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

loss_history = LossHistory()
'''Fit models and use validation_split=0.1 '''
history_adam = model_adam.fit(X_train, Y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=0,
                            shuffle=True,
                            validation_split=0.1,
                            callbacks=[early_stopping,loss_history,checkpoint])


loss_adam = history_adam.history.get('loss')
acc_adam = history_adam.history.get('acc')
val_loss_adam = history_adam.history.get('val_loss')
val_acc_adam = history_adam.history.get('val_acc')

loss = loss_history.loss
acc  = loss_history.acc
val_loss = loss_history.val_loss
val_acc  = loss_history.val_acc

''' Visualize the loss and accuracy of both models'''
import matplotlib.pyplot as plt
plt.figure(0,figsize=(8,6))
plt.subplot(121)
plt.plot(range(len(loss_adam)), loss_adam,label='Training')
plt.plot(range(len(val_loss_adam)), val_loss_adam,label='Validation')
plt.title('Loss history returned by fit function')
plt.xlabel("epoch")
plt.legend(loc='upper left')
plt.subplot(122)
plt.plot(range(len(loss)), loss,label='Training')
plt.plot(range(len(val_loss)), val_loss,label='Validation')
plt.title('Loss history from Callbacks')
plt.xlabel("epoch")
plt.tight_layout()
plt.savefig('10_callback.png',dpi=300,format='png')
plt.close()
print('Result saved into 10_callback.png')