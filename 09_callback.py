''' Import theano and numpy '''
import theano
import numpy as np
execfile('00_readingInput.py')

''''''
from keras.callbacks import Callback
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []
    def on_batch_end(self, batch, logs={}):
        self.loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))

loss_history=LossHistory()

from keras.callbacks import LearningRateScheduler
 
# learning rate schedule
def step_decay(epoch):
    initial_lrate = 0.1
    lrate = initial_lrate * 0.999
    return lrate

lrate = LearningRateScheduler(step_decay)
