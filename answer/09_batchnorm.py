#coding=utf-8
''' Import library '''
# from __future__ import print_function
# from past.builtins import execfile
# execfile('00_readingInput.py')
import numpy as np
exec(open("00_readingInput.py").read())

''' set the size of mini-batch and number of epochs'''
batch_size = 32
epochs = 100

''' Import keras to build a DL model '''
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import BatchNormalization
from keras.optimizers import SGD, Adam, RMSprop, Adagrad

def create_model_withDrp_bn(ratio_dropout, bn = False):
    print('Building current best model with Dropout = %g' % ratio_dropout)
    model_adam = Sequential()
    model_adam.add(Dense(128, input_dim=200))
    model_adam.add(Activation('relu'))
    if bn:
        model_adam.add(BatchNormalization())
    model_adam.add(Dropout(ratio_dropout))
    
    model_adam.add(Dense(256))
    if bn:
        model_adam.add(BatchNormalization())
    model_adam.add(Activation('relu'))
    model_adam.add(Dropout(ratio_dropout))
    
    model_adam.add(Dense(5))
    model_adam.add(Activation('softmax'))
    ##
    model_adam.compile(loss= 'categorical_crossentropy',
                    optimizer='Adam',
                    metrics=['accuracy'])
    return model_adam

model_adam_drp40 = create_model_withDrp_bn(0.4, bn = False)
model_adam_bn = create_model_withDrp_bn(0, bn = True)
model_adam_bn_ks = create_model_withDrp_bn(0, bn = True)
model_adam_drp40_bn = create_model_withDrp_bn(0.4, bn = True)

'''Fit models and use validation_split=0.1 '''
history_adam_drp40 = model_adam_drp40.fit(X_train, Y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=2,
                            shuffle=True,
                            validation_split=0.1)

history_adam_bn = model_adam_bn.fit(X_train, Y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=2,
                            shuffle=True,
                            validation_split=0.1)

history_adam_drp40_bn = model_adam_drp40_bn.fit(X_train, Y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=2,
                            shuffle=True,
                            validation_split=0.1)

batch_size = 2
history_adam_bn_ks = model_adam_bn_ks.fit(X_train, Y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=2,
                            shuffle=True,
                            validation_split=0.1)



def get_result(history_model):
    train_loss = history_model.history.get('loss')
    train_acc = history_model.history.get('acc')
    valid_loss = history_model.history.get('val_loss')
    valid_acc = history_model.history.get('val_acc')
    return train_loss, train_acc, valid_loss, valid_acc

loss_adam_drp40, acc_adam_drp40, val_loss_adam_drp40, val_acc_adam_drp40 = get_result(history_adam_drp40)
loss_adam_bn, acc_adam_bn, val_loss_adam_bn, val_acc_adam_bn = get_result(history_adam_bn)
loss_adam_bn_ks, acc_adam_bn_ks, val_loss_adam_bn_ks, val_acc_adam_bn_ks = get_result(history_adam_bn_ks)
loss_adam_drp40_bn, acc_adam_drp40_bn, val_loss_adam_drp40_bn, val_acc_adam_drp40_bn = get_result(history_adam_drp40_bn)

''' Visualize the loss and accuracy of both models'''
import matplotlib.pyplot as plt
skp = 10
plt.figure(0)
plt.plot(range(len(loss_adam_drp40)), loss_adam_drp40,label='Training_drp40')
plt.plot(range(len(val_loss_adam_drp40)), val_loss_adam_drp40,label='Validation_drp40')

plt.plot(range(len(loss_adam_drp40_bn)), loss_adam_drp40_bn,label='Training_drp40_bn')
plt.plot(range(len(val_loss_adam_drp40_bn)), val_loss_adam_drp40_bn,label='Validation_drp40_bn')

plt.plot(range(len(loss_adam_bn)), loss_adam_bn,label='Training_bn')
plt.plot(range(len(val_loss_adam_bn)), val_loss_adam_bn,label='Validation_bn')

plt.plot(range(len(loss_adam_bn_ks)), loss_adam_bn_ks,label='Training_bn (bz=2)')
plt.plot(range(len(val_loss_adam_bn_ks)), val_loss_adam_bn_ks,label='Validation_bn (bz=2)')


plt.legend(loc = 2, ncol = 2, fontsize = 8)
plt.title('Loss')
plt.savefig('09_bn_vs_dropout_loss.png',dpi=300,format='png')
plt.close()

plt.plot(range(len(acc_adam_drp40)), 
         acc_adam_drp40,
         label='Training_drp40')
plt.plot(range(len(val_acc_adam_drp40)), 
         val_acc_adam_drp40,
         label='Validation_drp40')
plt.plot(range(len(acc_adam_drp40_bn)), 
         acc_adam_drp40_bn,
         label='Training_drp40_bn')
plt.plot(range(len(val_acc_adam_drp40_bn)), 
         val_acc_adam_drp40_bn,
         label='Validation_drp40_bn')
plt.plot(range(len(acc_adam_bn)), 
         acc_adam_bn,
         label='Training_bn')
plt.plot(range(len(val_acc_adam_bn)), 
         val_acc_adam_bn,
         label='Validation_bn')
plt.plot(range(len(acc_adam_bn_ks)), 
         acc_adam_bn_ks,
         label='Training_bn (bz=2)')
plt.plot(range(len(val_acc_adam_bn_ks)), 
         val_acc_adam_bn_ks,
         label='Validation_bn (bz=2)')

plt.legend(loc = 4, ncol = 2, fontsize = 8)
plt.title('Accuracy')
plt.xlabel("epoch")
plt.tight_layout()
plt.savefig('09_bn_vs_dropout_accuracy.png',dpi=300,format='png')
plt.close()
