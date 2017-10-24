"""
LSTM for time series classification

This model takes in time series and class labels.
The LSTM models the time series. A fully-connected layer
generates an output to be classified with Softmax
"""

import numpy as np
import tensorflow as tf  #TF 1.1.0rc1
tf.logging.set_verbosity(tf.logging.ERROR)
#import matplotlib.pyplot as plt
from tsc_model import Model,sample_batch,load_data#,check_test
from scipy import sparse
import math

def pad(A, length):
    arr = np.zeros(length)
    arr[:len(A)] = A
    return arr

#these directories
direc = '/home/bmocialov/LSTM_tsc/UCR_TS_Archive_2015'
summaries_dir = '/home/bmocialov/LSTM_tsc'

"""Load the data"""
ratio = np.array([0.8,0.9]) #Ratios where to split the training and validation set
X_train,X_val,X_test,y_train,y_val,y_test = load_data(direc,ratio,dataset='DutchSL')
#print len(X_train)
#N = len(X_train)
#sl = 1
#print X_train
X_test = np.asarray(X_test)
X_test = np.atleast_2d(X_test)
X_train = np.asarray(X_train)
X_train = np.atleast_2d(X_train)
print X_train.shape, X_test.shape
max_size = -1
for item in X_train:
  #print item, len(item)
  for item2 in item:
    #print item2, len(item2)
    if len(item2) > max_size:
      max_size = len(item2)

#print len(X_test)
for item in X_test:
  for item2 in item:
    if len(item2) > max_size:
      max_size = len(item2)
print "max_size", max_size
max_size = 500
N,sl = X_train.shape
sl = max_size
#print "N,sl", N, sl
#print y_train
num_classes = len(np.unique(y_train))

#print "data loaded"

"""Hyperparamaters"""
batch_size = 80
max_iterations = 300000
dropout = 0.8
config = {    'num_layers' :    5,               #number of layers of stacked RNN's
              'hidden_size' :   120,             #memory cells in a layer
              'max_grad_norm' : 5,             #maximum gradient norm during training
              'batch_size' :    batch_size,
              'learning_rate' : .005,
              'sl':             sl,
              'num_classes':    num_classes}



epochs = np.floor(batch_size*max_iterations / N)
print('Train %.0f samples in approximately %d epochs' %(N,epochs))
#print "1"
#Instantiate a model
model = Model(config)
#print "2"
"""Session time"""
sess = tf.Session() #Depending on your use, do not forget to close the session
writer = tf.summary.FileWriter(summaries_dir, sess.graph)  #writer for Tensorboard
sess.run(model.init_op)

cost_train_ma = -np.log(1/float(num_classes)+1e-9)  #Moving average training cost
acc_train_ma = 0.0
try:
  for i in range(max_iterations):
    #print "sample batch"
    X_batch, y_batch = sample_batch(X_train,y_train,batch_size)
    #max_length = -1
    #reference_shape  = (0,0)
    #for item in X_batch:
    #    print len(item)
    #    if len(item) > max_length:
    #        max_length = len(item)
    #        reference_shape = np.asarray(item).shape
    for idx,item in enumerate(X_batch):
        #print "doing stuff", idx
        #max_size = 20
        if len(item) > max_size:
          new_item = item[:max_size]
        else:
          new_item = pad(item, max_size) #np.zeros(reference_shape)
        #item = np.atleast_2d(np.asarray(item))
        #print "shape", np.shape(item)
        #new_item[:item.shape[0],:item.shape[1]] = item
        X_batch[idx] = new_item
    #X_batch_ = np.vstack([np.expand_dims(x, 0) for x in X_batch])
    
    #print "length of X_batch", len(X_batch)
    #for item in X_batch:
    #    print item
    #X_batch_ = np.vstack([np.expand_dims(x,0) for x in X_batch])
    #X_batch_ = np.reshape(X_batch_,(1, X_batch_.shape[0]))
    #X_batch_ = np.reshape(X_batch_, (-1, max_size))
    X_batch = np.vstack([np.expand_dims(x, 0) for x in X_batch])
    #print X_batch, y_batch
    #Next line does the actual training
    #print "model.input", model.input
    #print "run"
    cost_train, acc_train,_ = sess.run([model.cost,model.accuracy, model.train_op],feed_dict = {model.input: X_batch,model.labels: y_batch,model.keep_prob:dropout})
    cost_train_ma = cost_train_ma*0.99 + cost_train*0.01
    acc_train_ma = acc_train_ma*0.99 + acc_train*0.01
    if i%100 == 1:
    #Evaluate validation performance
      X_batch, y_batch = sample_batch(X_val,y_val,batch_size)
      for idx,item in enumerate(X_batch):
          #print "doing stuff", idx
          #max_size = 20
          if len(item) > max_size:
            new_item = item[:max_size]
          else:
            new_item = pad(item, max_size) #np.zeros(reference_shape)
          #item = np.atleast_2d(np.asarray(item))
          #print "shape", np.shape(item)
          #new_item[:item.shape[0],:item.shape[1]] = item
          X_batch[idx] = new_item

      X_batch_ = np.vstack([np.expand_dims(x, 0) for x in X_batch])
      cost_val, summ,acc_val = sess.run([model.cost,model.merged,model.accuracy],feed_dict = {model.input: X_batch_, model.labels: y_batch, model.keep_prob:1.0})
      print('At %5.0f/%5.0f: COST %5.3f/%5.3f(%5.3f) -- Acc %5.3f/%5.3f(%5.3f)' %(i,max_iterations,cost_train,cost_val,cost_train_ma,acc_train,acc_val,acc_train_ma))
      #Write information to TensorBoard
      writer.add_summary(summ, i)
      writer.flush()
except KeyboardInterrupt:
  pass
  
epoch = float(i)*batch_size/N
print('Trained %.1f epochs, accuracy is %5.3f and cost is %5.3f'%(epoch,acc_val,cost_val))

#now run in your terminal:
# $ tensorboard --logdir = <summaries_dir>
# Replace <summaries_dir> with your own dir



