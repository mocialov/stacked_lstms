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
from tsc_model import Model,sample_batch,load_data, load_data_si#,check_test
from scipy import sparse
import math
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM, TimeDistributed
from keras.utils.np_utils import to_categorical
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
import math
from keras.models import model_from_json
from sklearn.metrics import confusion_matrix
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from keras import optimizers
from keras import regularizers


epinthesis_flag = False
load_flag = False
load_image_flag = False

model_file_name = "model_1_si"
features_num=174 #384,174,222
dataset = "1_classes_without_face_shuffled"
epoches = 500

labels_dict = {0: 2020, 1: 606, 2: 808, 3: 1818, 4: 909, 5: 808, 6: 606, 7: 1212, 8: 707, 9: 3434}

def pad(A, length):
    arr = np.zeros(length)
    arr[:len(A)] = A
    return arr

def chunks(l, n):
  for i in xrange(0, len(l), n):
    yield l[i:i+n]

def create_class_weight(labels_dict,mu=0.15):
    total = np.sum(labels_dict.values())
    keys = labels_dict.keys()
    class_weight = dict()

    for key in keys:
        score = math.log(mu*total/float(labels_dict[key]))
        class_weight[key] = score if score > 1.0 else 1.0

    return class_weight

#these directories
direc = '/home/bmocialov/LSTM_tsc/UCR_TS_Archive_2015'
summaries_dir = '/home/bmocialov/LSTM_tsc'

"""Load the data"""
ratio = np.array([0.8,0.9]) #Ratios where to split the training and validation set
#X_train,X_val,X_test,y_train,y_val,y_test = load_data(direc,ratio,dataset='DutchSL',postfix=dataset)
X_train,X_val,X_test,y_train,y_val,y_test = load_data_si(direc,dataset="DutchSL",postfix=dataset)


#X_train = X_train[0:len(X_train)/10]
#X_val = X_val[0:len(X_val)/10]
#X_test = X_test[0:len(X_test)/10]
#y_train = y_train[0:len(y_train)/10]
#y_val = y_val[0:len(y_val)/10]
#y_test = y_test[0:len(y_test)/10]

#print len(X_train)
#N = len(X_train)
#sl = 1
#print X_train
X_test = np.asarray(X_test)
X_test = np.atleast_2d(X_test)
X_train = np.asarray(X_train)
X_train = np.atleast_2d(X_train)
X_val = np.atleast_2d(X_val)

sizes=[]
print X_train.shape, X_test.shape, X_val.shape
max_size = -1
for item in X_train:
  #print item, len(item)
  for item2 in item:
    #print len(item2)
    if len(item2) > max_size:
      max_size = len(item2)
      sizes.append(max_size)

#print len(X_test)
for item in X_test:
  for item2 in item:
    #print len(item2)
    if len(item2) > max_size:
      max_size = len(item2)
      sizes.append(max_size)

for item in X_val:
  for item2 in item:
    #print len(item2)
    if len(item2) > max_size:
      max_size = len(item2)
      sizes.append(max_size)
print sizes
#print "max_size", max_size
#max_size = 10
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



for idx1, item in enumerate(X_train):
  for idx2, item2 in enumerate(item):
    if len(item2) < max_size:
      X_train[idx1][idx2] = pad(item2, max_size)
    else:
      X_train[idx1][idx2] = X_train[idx1][idx2][:max_size]

for idx1, item in enumerate(X_test):
  for idx2, item2 in enumerate(item):
    if len(item2) < max_size:
      X_test[idx1][idx2] = pad(item2, max_size)
    else:
      X_test[idx1][idx2] = X_test[idx1][idx2][:max_size]

for idx1, item in enumerate(X_val):
  for idx2, item2 in enumerate(item):
    if len(item2) < max_size:
      X_val[idx1][idx2] = pad(item2, max_size)
    else:
      X_val[idx1][idx2] = X_val[idx1][idx2][:max_size]


for idx, item in enumerate(X_train[0]):
  #print len(item)
  X_train[0][idx] = np.array_split(item, features_num)

for idx, item in enumerate(X_test[0]):
  #print len(item)
  X_test[0][idx] = np.array_split(item, features_num)

for idx, item in enumerate(X_val[0]):
  #print len(item)
  X_val[0][idx] = np.array_split(item, features_num)

X_train = X_train[0]
X_train = np.array(X_train)
X_train = np.vstack([np.expand_dims(x, 0) for x in X_train])
y_train_copy = y_train
y_train = np.eye(10)[y_train]

X_test = X_test[0]
X_test = np.array(X_test)
X_test = np.vstack([np.expand_dims(x, 0) for x in X_test])
y_test_copy = y_test
y_test = np.eye(10)[y_test]

X_val = X_val[0]
X_val = np.array(X_val)
X_val = np.vstack([np.expand_dims(x, 0) for x in X_val])
y_val_copy = y_val
y_val = np.eye(10)[y_val]

#print X_train.shape
#print X_test.shape
#print X_val.shape

#X_train = np.vstack((X_train,X_test))
#X_train = np.vstack((X_train,X_val))
#y_train = np.vstack((y_train, y_test))
#y_train = np.vstack((y_train, y_val))
#sys.exit(0)
print "training data shape", X_train.shape
#print y_train.shape
print "validation data shape", X_val.shape
#print y_val.shape
print "testing data shape", X_test.shape

print "classes representatives in training dataset", Counter(np.asarray(y_train_copy).flatten())
print "Classes representatives in validation dataset", Counter(np.asarray(y_val_copy).flatten())
print "Classes representatives in testing dataset", Counter(np.asarray(y_test_copy).flatten())

#model = Sequential()
#model.add(Conv1D(64,3, activation='relu', input_shape=(165,58)))
#model.add(Conv1D(64,3, activation='relu'))
#model.add(MaxPooling1D(3))
#model.add(Conv1D(128,3, activation='relu'))
#model.add(Conv1D(128,3, activation='relu'))
#model.add(GlobalAveragePooling1D())
#model.add(Dropout(0.5))
#model.add(Dense(14, activation='sigmoid'))
#model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])



if(epinthesis_flag):
  import csv
  epinthesis_test = []
  datafile = open('UCR_TS_Archive_2015/DutchSL/epinthesis.csv', 'r')
  datareader = csv.reader(datafile)
  data = []
  counter = 0
  for idx1,row in enumerate(datareader):
    counter +=1
    #print [float(i) for i in list(row)]
    #sys.exit(0)
    # I split the input string based on the comma separator, and cast every elements into a float
    epinthesis_test.append( [float(i) for i in list(row)] )
    #if idx1 == 999:
    #  break
  #print epinthesis_test
  #print "been hefre", counter

  #from numpy import genfromtxt
  #epinthesis_data = genfromtxt('epinthesis_segments.csv', delimiter=',')
  #epinthesis_data = np.array(epinthesis_data)
  #epinthesis_test = epinthesis_test / max(abs(max(epinthesis_test)), abs(min(epinthesis_test)))

  epinthesis_test = np.asarray(epinthesis_test)
  epinthesis_test = np.atleast_2d(epinthesis_test)
  #epinthesis_test = [epinthesis_test]
  #print epinthesis_test
  #print ""
  #print epinthesis_test[0] # [ [], [], [], ...]
  for idx1, item in enumerate(epinthesis_test):
    counter = 0
    for idx2, item2 in enumerate(epinthesis_test[idx1]):
      counter += 1
      #print len(item2)
      if len(item2) < max_size:
        epinthesis_test[idx1][idx2] = pad(item2, max_size)
      else:
        epinthesis_test[idx1][idx2] = epinthesis_test[idx1][idx2][:max_size]
  #print epinthesis_test.shape
  #print epinthesis_test[0].shape
  #print epinthesis_test[0][0].shape
  #print "beeen there ", counter
  #for idx,item in enumerate(epinthesis_test):
  #  if len(item) < max_size:
  #    epinthesis_test[idx] = pad(item, max_size)
  #  else:
  #    epinthesis_test[idx] = epinthesis_test[idx][:max_size]

  #print epinthesis_test
  for idx, item in enumerate(epinthesis_test[0]):
    epinthesis_test[0][idx] = np.array_split(item, features_num)
  #print epinthesis_test

  epinthesis_test = epinthesis_test[0]
  epinthesis_test = np.array(epinthesis_test)
  epinthesis_test = np.vstack([np.expand_dims(x, 0) for x in epinthesis_test])


  #for idx, item in enumerate(epinthesis_test):
  #  #print len(item)
  #  epinthesis_test[idx] = np.array_split(item, 174)

  #for item in converted:
  #print len(converted)
  #print len(converted[0])
  
  json_file = open(model_file_name+".json", "r")
  loaded_model_json = json_file.read()
  json_file.close()
  loaded_model = model_from_json(loaded_model_json)
  loaded_model.load_weights(model_file_name+".h5")
  loaded_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
  predicted = loaded_model.predict_proba(epinthesis_test)
  
  print len(predicted)
  for item in predicted:
    print np.argmax(item), np.max(item)

  sys.exit(0)

if(load_flag):
  json_file = open(model_file_name+".json", "r")
  loaded_model_json = json_file.read()
  json_file.close()
  loaded_model = model_from_json(loaded_model_json)
  loaded_model.load_weights(model_file_name+".h5")

  loaded_model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
  #loaded_model.predict(X_test)
  score = loaded_model.evaluate(X_test, y_test, batch_size=32)
  print "final score", score
  predicted = loaded_model.predict_classes(X_test)
  
  #print len(predicted), len(y_test_copy)
  confusion = confusion_matrix(y_test_copy, predicted)
  #print confusion
  confusion_norm = confusion.astype('float') / confusion.sum(axis=1)[:, np.newaxis]
  #print confusion_norm
  if (load_image_flag):
    labels = ["ape","building","electricity","handwriting","look","poet","rain","run","shake","tram","ball","binoculars","bird","birdcage","inside","not","ready","rope","same","search"]#,"and","apartment","climb","corner","how","hurry","line","old","pipe","thinking"]
    #plt.rcParams['ytick.labelsize'] = small
    fig = plt.figure()
    #fig = plt.figure(figsize=(60,60))
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion_norm)
    fig.colorbar(cax)
    #fig.subplots_adjust(top=0.02)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation='vertical')
    ax.set_yticklabels(labels, rotation='horizontal')
    #plt.subplots_adjust(left=1, right=1, top=1, bottom=1)
    fig.tight_layout()
    #plt.matshow(confusion_norm)
    #plt.colorbar()
    #plt.ylabel('True Label')
    #plt.xlabel('Predicted Label')
    plt.savefig('confusion_20_with_face_reduced.jpg')


  sys.exit(0)

model = Sequential()

#model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2, activity_regularizer=regularizers.l1_l2(0.01), kernel_regularizer=regularizers.l1_l2(0.01), recurrent_regularizer=regularizers.l1_l2(0.01), return_sequences=True, input_shape=(features_num, 55)))  # returns a sequence of vectors of dimension 32   input shape = 384|174
#model.add(Dropout(0.2))
#model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2, activity_regularizer=regularizers.l1_l2(0.01), kernel_regularizer=regularizers.l1_l2(0.01), recurrent_regularizer=regularizers.l1_l2(0.01), return_sequences=True))  # returns a sequence of vectors of dimension 32
#model.add(Dropout(0.2))
#model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2, activity_regularizer=regularizers.l1_l2(0.01), kernel_regularizer=regularizers.l1_l2(0.01), recurrent_regularizer=regularizers.l1_l2(0.01)))  # return a single vector of dimension 32
#model.add(Dropout(0.2))
#model.add(Dense(10, activation='softmax'))

model.add(LSTM(32, return_sequences=True, input_shape=(features_num, 55)))  # returns a sequence of vectors of dimension 32   input shape = 384|174
model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
model.add(LSTM(32))  # return a single vector of dimension 32
model.add(Dense(10, activation='softmax'))

opt=optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(loss='categorical_crossentropy',#'categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])


#checkpoint
filepath=model_file_name+".best.hdf5"
print "save into", filepath
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

class_weights = create_class_weight(labels_dict)
#model.fit(X_train, y_train, validation_split=0.2, batch_size=16, epochs=1000)
model.fit(X_train, y_train, batch_size=32, callbacks=callbacks_list, epochs=epoches, shuffle=True, validation_data=(X_val, y_val), class_weight = class_weights)
score = model.evaluate(X_test, y_test, batch_size=32)
print "final score", score

model_json = model.to_json()
with open(model_file_name+".json", "w") as json_file:
    json_file.write(model_json)
model.save_weights(model_file_name+".h5")

sys.exit(0)


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



