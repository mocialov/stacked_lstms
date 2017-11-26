# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 12:46:39 2017

@author: rob
"""

"""
LSTM for time series classification
Made: 30 march 2016

This model takes in time series and class labels.
The LSTM models the time series. A fully-connected layer
generates an output to be classified with Softmax
"""

import sys
import csv
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops
from tensorflow.contrib.rnn.python.ops import rnn_cell
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.rnn.python.ops import core_rnn

def load_data_si(direc, dataset, postfix="", prefix="signer_independent_"):
  datadir = direc + '/' + dataset + '/' + prefix + dataset
  
  data_train = []
  data_validation = []
  data_testing = []

  datafile = open(datadir+'_TRAIN_'+postfix, 'r')
  datareader = csv.reader(datafile)
  for idx1,row in enumerate(datareader):
    data_train.append( [float(i) for i in list(row)] )

  datafile = open(datadir+'_VALIDATION_'+postfix, 'r')
  datareader = csv.reader(datafile)
  for idx1,row in enumerate(datareader):
    data_validation.append( [float(i) for i in list(row)] )

  datafile = open(datadir+'_TESTING_'+postfix, 'r')
  datareader = csv.reader(datafile)
  for idx1,row in enumerate(datareader):
    data_testing.append( [float(i) for i in list(row)] )



  data_train = np.asarray(data_train)
  data_validation = np.asarray(data_validation)
  data_testing = np.asarray(data_testing)



  ind_train = np.random.permutation(len(data_train))
  ind_validation = np.random.permutation(len(data_validation))
  ind_testing = np.random.permutation(len(data_testing))


  
  X_train = []
  for i in ind_train[:]:
    X_train.append(data_train[[i]][0][1:])
  
  X_val = []
  for i in ind_validation[:]:
    X_val.append(data_validation[[i]][0][1:])
  
  X_test = []
  for i in ind_testing[:]:
    X_test.append(data_testing[[i]][0][1:])
  
  y_train = []
  for i in ind_train[:]:
    y_train.append(int(data_train[[i]][0][0]))
  
  y_val = []
  for i in ind_validation[:]:
    y_val.append(int(data_validation[[i]][0][0]))
  
  y_test = []
  for i in ind_testing[:]:
    y_test.append(int(data_testing[[i]][0][0]))#
  
  return X_train,X_val,X_test,y_train,y_val,y_test


def load_data(direc,ratio,dataset,postfix="",prefix="signer_independent_"):
  """Input:
  direc: location of the UCR archive
  ratio: ratio to split training and testset
  dataset: name of the dataset in the UCR archive"""
  datadir = direc + '/' + dataset + '/' + prefix + dataset
  #data_train = np.genfromtxt(datadir+'_TRAIN',delimiter=',')
  #data_test_val = np.genfromtxt(datadir+'_TEST',delimiter=',')
  
  data_train = []
  data_test_val = []
  #print datadir+"_TRAIN_30_classes"
  datafile = open(datadir+'_TRAIN_'+postfix, 'r')
  datareader = csv.reader(datafile)
  data = []
  for idx1,row in enumerate(datareader):
    #print [float(i) for i in list(row)]
    #sys.exit(0)
    # I split the input string based on the comma separator, and cast every elements into a float
    data_train.append( [float(i) for i in list(row)] )
    #if idx1 == 999:
    #  break
  
  datafile = open(datadir+'_TESTING_'+postfix, 'r')
  datareader = csv.reader(datafile)
  data = []
  for idx2, row in enumerate(datareader):
    # I split the input string based on the comma separator, and cast every elements into a float
    data_test_val.append( [ float(i) for i in list(row) ] )
    #if idx2 == 1000:
    #  break
  
  #print "train", data_train
  #print "test", data_test_val

  DATA = np.concatenate((data_train,data_test_val),axis=0)
  DATA = np.asarray(DATA)
  
  N = DATA.shape[0]
  #print N
  ratio = (ratio*N).astype(np.int32)
  print len(data_train), len(data_test_val)
  #print ratio
  #sys.exit(0)
  ind = np.random.permutation(N)
  #print "data", DATA[ind[:ratio[0]]][0][1:]

  #print "ratio", ratio[0], "ind", ind
  #X_train = DATA[ind[:ratio[0]],1:]
  X_train = []
  for i in ind[:ratio[0]]:
    X_train.append(DATA[[i]][0][1:])
  #with open('temp_train.csv', 'wb') as csvfile:
  #  writer = csv.writer(csvfile)
  #  writer.writerows(X_train)
  #X_val = DATA[ind[ratio[0]:ratio[1]],1:]
  X_val = []
  for i in ind[ratio[0]:ratio[1]]:
    X_val.append(DATA[[i]][0][1:])
  #with open('temp_val.csv', 'wb') as csvfile:
  #  writer = csv.writer(csvfile)
  #  writer.writerows(X_val)
  #X_test = DATA[ind[ratio[1]:],1:]
  X_test = []
  for i in ind[ratio[1]:]:
    X_test.append(DATA[[i]][0][1:])
  #with open('temp_test.csv', 'wb') as csvfile:
  #  writer = csv.writer(csvfile)
  #  writer.writerows(X_test)
  # Targets have labels 1-indexed. We subtract one for 0-indexed
  #y_train = DATA[ind[:ratio[0]],0:]-1
  y_train = []
  for i in ind[:ratio[0]]:
    y_train.append(int(DATA[[i]][0][0]))
  #y_val = DATA[ind[ratio[0]:ratio[1]],0]-1
  #with open('temp_train_y.csv', 'wb') as csvfile:
  #  writer = csv.writer(csvfile)
  #  writer.writerows(y_train)
  y_val = []
  for i in ind[ratio[0]:ratio[1]]:
    y_val.append(int(DATA[[i]][0][0]))
  #y_test = DATA[ind[ratio[1]:],0:]-1
  #with open('temp_val_y.csv', 'wb') as csvfile:
  #  writer = csv.writer(csvfile)
  #  writer.writerows(y_val)
  y_test = []
  for i in ind[ratio[1]:]:
    y_test.append(int(DATA[[i]][0][0]))#
  #with open('temp_test_y.csv', 'wb') as csvfile:
  #  writer = csv.writer(csvfile)
  #  writer.writerows(y_test)


  #for i in X_train:
  #print len(X_train), len(X_val), len(X_test), y_train, y_val, y_test
  return X_train,X_val,X_test,y_train,y_val,y_test


def sample_batch(X_train,y_train,batch_size):
  """ Function to sample a batch for training"""
  #N,data_len = X_train.shape
  #print "N:", N
  #print "data_len:", data_len
  #print "train:", X_train
  X_train = np.asarray(X_train)
  y_train = np.asarray(y_train)
  X_train = np.atleast_2d(X_train)
  #print "sample_bach", X_train.shape
  data_len,N = X_train.shape
  #print "hello"
  #print X_train
  #print "shape: ", X_train.shape, N, data_len
  ind_N = np.random.choice(N,batch_size,replace=False)
  #print X_train
  X_batch = X_train[0][ind_N]
  #for i in X_batch:
  #  print i
  #print "batch: ", X_batch
  #print "index", ind_N, "at: ", y_train
  y_batch = y_train[ind_N]
  return X_batch,y_batch


class Model():
  def __init__(self,config):
    
    num_layers = config['num_layers']
    #print num_layers
    hidden_size = config['hidden_size']
    max_grad_norm = config['max_grad_norm']
    self.batch_size = config['batch_size']
    sl = config['sl']
    #print "sl", sl
    learning_rate = config['learning_rate']
    num_classes = config['num_classes']
    """Place holders"""
    self.input = tf.placeholder(tf.float32, [None, sl], name = 'input')
    self.labels = tf.placeholder(tf.int64, [None], name='labels')
    self.keep_prob = tf.placeholder("float", name = 'Drop_out_keep_prob')
    #print "here0"
    with tf.name_scope("LSTM_setup") as scope:
      def single_cell():
        return tf.contrib.rnn.DropoutWrapper(LSTMCell(hidden_size),output_keep_prob=self.keep_prob)

      cell = tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(num_layers)])
      initial_state = cell.zero_state(self.batch_size, tf.float32)
    #print "here01"
    input_list = tf.unstack(tf.expand_dims(self.input,axis=2),axis=1)
    #print "here0000001"
    outputs,_ = core_rnn.static_rnn(cell, input_list, dtype=tf.float32)
    #print "here00000000001"
    output = outputs[-1]
    #print "here001"

    #Generate a classification from the last cell_output
    #Note, this is where timeseries classification differs from sequence to sequence
    #modelling. We only output to Softmax at last time step
    #print "here1"
    with tf.name_scope("Softmax") as scope:
      with tf.variable_scope("Softmax_params"):
        softmax_w = tf.get_variable("softmax_w", [hidden_size, num_classes])
        softmax_b = tf.get_variable("softmax_b", [num_classes])
      logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
      #Use sparse Softmax because we have mutually exclusive classes
      loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=self.labels,name = 'softmax')
      self.cost = tf.reduce_sum(loss) / self.batch_size
    with tf.name_scope("Evaluating_accuracy") as scope:
      correct_prediction = tf.equal(tf.argmax(logits,1),self.labels)
      self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
      h1 = tf.summary.scalar('accuracy',self.accuracy)
      h2 = tf.summary.scalar('cost', self.cost)

    print "running optimizer"
    """Optimizer"""
    with tf.name_scope("Optimizer") as scope:
      tvars = tf.trainable_variables()
      grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),max_grad_norm)   #We clip the gradients to prevent explosion
      optimizer = tf.train.AdamOptimizer(learning_rate)
      gradients = zip(grads, tvars)
      self.train_op = optimizer.apply_gradients(gradients)
      # Add histograms for variables, gradients and gradient norms.
      # The for-loop loops over all entries of the gradient and plots
      # a histogram. We cut of
      # for gradient, variable in gradients:  #plot the gradient of each trainable variable
      #       if isinstance(gradient, ops.IndexedSlices):
      #         grad_values = gradient.values
      #       else:
      #         grad_values = gradient
      #
      #       tf.summary.histogram(variable.name, variable)
      #       tf.summary.histogram(variable.name + "/gradients", grad_values)
      #       tf.summary.histogram(variable.name + "/gradient_norm", clip_ops.global_norm([grad_values]))

    #Final code for the TensorBoard
    self.merged = tf.summary.merge_all()
    self.init_op = tf.global_variables_initializer()
    print('Finished computation graph')



