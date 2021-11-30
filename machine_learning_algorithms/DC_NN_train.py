"""
This is the code for training and save the trained neural network after appling in o data saved on th database written
by MECwritter.py and stored in a postgres database

designed to calculate the predicted probability of the classification algorithim of he harmonics independantly then
calculte the most probably reaction mechanism by using bayes inference on all availble harmonics

Arthor; Luke Gundry 11/20
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import psycopg2
#from PIL import Image
import keras
import itertools

from shutil import copyfile # for copying the NN atchitecture

import tensorflow as tf
#from keras.backend.tensorflow_backend import set_session
# code timing
import time
import datetime

# import custom files
import NN_train_mod as NNt
import DC_neuralnetwork_mod as DC_NNt

# sets seed for testing stability
from numpy.random import seed
#seed(666)  # sets seed for numpy which is used for keras and theano
#tf.random.set_seed(666) # sets the seed for tensorflow

print("start")
tic = time.time()

serverdata = ["DC_db_kf"]       # server data
imagedimensions = [100,100]#[224,224]#    # Current immage dimensions we are using for the
Datsetbol = True  # Set to False for generator set to True for Dataset
flipimages = True

testratio = 0.2 # training ratio data train = 0.2 = 1/5
trainratio = 1 - testratio

# put in frequency and labels for all experments
ACcomp = [0,0]  # AC component where
models = ["E","EC1st","EC2nd"]

# sets up model numbers
model_numcode = {}
for i in range(len(models)):
    model_numcode.update({models[i]:i})

# Machine learningparameters
dnnHLNodeNumber = 70
dnnLearningRate = 1.0e-4
n_labels = len(models)
dnnEpochs = 5 # dnnEpochs = 30
dnnBatch = 42 # 24
noise = 0.00 # percenage of max current noise 0 mean none

##tested filter paths
name = "3_3_3_3"
modeltype = "gareth" # wide or deep or gareth
activtype = "relu" # activation of varible layers (relu or tanh)
freqused = str(ACcomp[1])

#Number of cpus that are used to fill the batch size
cpu_number = 9

# creates a generic output file for the NN metadata/outputs
filename = NNt.genericoutpufile("CNN",name,modeltype,freqused,activtype) # input is name of file

copyfile("DC_NN_train.py",filename + "/NN_train_COPY.py") # this copy the files to output
copyfile("DC_neuralnetwork_mod.py",filename + "/NN_train_mod_COPY.py")

#"""GET EXPsetrow THAT RELATE TO THE ABOVE AC SINE PROPERTIES AND  MODEL""" #exp = [[expsetrow,reactionmech],...]
exp_class = NNt.EXPsetrow_collector(serverdata,models,ACcomp)

exp_data = NNt.EXPsetrow_dcdata(serverdata,exp_class)

# create list of form react_class = [[[ReactionID,Reactionmech]]] for each reac mech
react_class = NNt.ReactID_collector(serverdata,exp_class)

# count number for each reaction model and check if same
Narray = NNt.Narray_count(react_class)

# split and shuffle react_class for each independant reactionMech
#print("ony taking one 4 for ease of practise")
#y = []
#for i in range(len(models)):
#    x = react_class[i][::4]
#    y.append(x)
#react_class = y

# train data is for train NN, testdata is for testing NN
testdata, traindata, Ntest, Ntrain = NNt.suffle_split(testratio,Narray, react_class)

# train data via a progressive loading technique such as .flow based method where batch_size = 24
# NEED A GENERATOR need to be put in a big loop for the harmonics harmnum = harmonic we are training it on
batchsize = dnnBatch # just doing one at a time as im lazy needs to be extended

# below fix has been corrected
"""if flipimages: # if we doing top and bottom
    imagedimensions[0] = int(0.5*imagedimensions[0])
    imagedimensions[1] = int(2 * imagedimensions[1])"""

if modeltype == "gareth":
    n_channels = 1
elif modeltype == "gareth3N" or "deep3N":
    n_channels = 3
else:
    n_channels = 4

#settings for the neural network
params = {'dim': (imagedimensions[0],imagedimensions[1]),
          'batchsize': dnnBatch,
          'n_classes': len(models), # number of training models
          'n_channels': n_channels,      # the number of filters present
          'serverdata':serverdata ,
          'model_numcode':model_numcode,
          'cpu_number':cpu_number,
          'exp_data':exp_data,
          'modeltype':modeltype,
          "noise": noise}

# for clearing up CPU
keras.backend.clear_session()

for harmnum in [0]:#range(0,9):#range(0,10): # goes from harmonics 0 - 8

    intertic = time.time()
    # extracts the reaction IDs to format for processing
    harmtrain, harmtrain_mech, harmtrain_ID = DC_NNt.DC_NN_setter(traindata, harmnum)  # training data
    harmtest, harmtest_mech, harmtest_ID = DC_NNt.DC_NN_setter(testdata,harmnum) # testing data for validation

    # this is here o set up the batch information on train
    #harmtrain = NNt.NN_batchsetter(harmtrain, dnnBatch)
    #print(harmtrain)

    #Need something to collect all the training data


    # this will be need to be saved to metadata
    for mechs in models:
        print(mechs, end=': ')
        print(harmtrain_mech.count(mechs))

    CNNtrain = DC_NNt.Machine_learning_class(harmtrain,harmnum, **params)#harmnum,batchsize,imagedimensions,serverdata)

    print("need somehing different for the validation data")
    trainNumber = dnnBatch * 10
    #validationdata = NNt.FTACV_CNN_trainer(harmtest, trainNumber, 1, (imagedimensions[0],imagedimensions[1]), serverdata, harmnum,
    #                                       model_numcode)
    CNNtest = DC_NNt.Machine_learning_class(harmtest, harmnum, **params)
    # set up the generators for training and validation (testing)
    filtersize1 = (1, 1)
    filtersize3 = (3, 3)
    filtersize5 = (5, 5)
    filtersize7 = (7,7) # max filter size, try
    filtersize9 = (9, 9)
    filtersize11 = (11, 11)
    filtersize13 = (13, 13)

    filterlist = [filtersize3,filtersize5,filtersize7,filtersize9,filtersize11,filtersize13]
    filtername = ["_3","_5","_7","_9","_11","_13"]
    filterdic = []
    filternamedic = []

    # this is here for filter testing
    for x in name.split("_"):
        #print(x)
        j = int((int(x)-1)/2 - 1)
        filterdic.append(filterlist[j])

    # build the model using sequential layers
    #model = Sequential()
    # 3 sequential 2D convolution layers followed by Rectified Linear Unit (Relu)
    #   activation function and a 2D max pooling layer (1, 100, 100, 1) (#num,
    print("pooop1")
    #model = NNt.widemodel()
    spatialdropout = 0.5
    if modeltype == "deep" or modeltype == "deep3N":
        model = DC_NNt.deepmodel_DC(imagedimensions,spatialdropout,dnnHLNodeNumber,n_labels,filterdic,activtype,n_channels)
    elif modeltype == "wide":
        model = DC_NNt.widemodel(imagedimensions, spatialdropout, dnnHLNodeNumber, n_labels, filterdic,activtype)
    elif modeltype == "gareth":
        model = DC_NNt.garethmodel(imagedimensions,dnnHLNodeNumber,n_labels)
    elif modeltype == "gareth4N":
        model = DC_NNt.garethmodel4N(imagedimensions, dnnHLNodeNumber, n_labels)
    elif modeltype == "gareth3N":

        model = DC_NNt.garethmodel3N(imagedimensions, dnnHLNodeNumber, n_labels)
    print("pooop")
    tf.keras.utils.plot_model(model, to_file=filename + "/model_"+name+".png", show_shapes=True)

    adam_opt = keras.optimizers.Adam(lr=dnnLearningRate)

    model.compile(optimizer=adam_opt, loss='categorical_crossentropy', metrics=['accuracy'])

    print(model.summary())
    # For simplicity
    print("model summary")
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print(tf.keras.mixed_precision.global_policy())

    history = model.fit(x=CNNtrain, batch_size=dnnBatch, epochs = dnnEpochs, validation_data = CNNtest,
              validation_batch_size = dnnBatch,max_queue_size=20)# use_multiprocessing=F, workers=cpu_number)

    NNt.accuracyplot(history,harmnum,filename)

    modelname = filename + "/testmodel_" + str(harmnum) +".model"
    model.save(modelname)
    # train the bloddy thing

    NNt.genericfitingdata(filename, harmnum, intertic,model_numcode,models,harmtrain_mech)

    # clear all neural network data from system and gpu
    del model  # for avoid any trace on aigen
    #tf.reset_default_graph()  # for being sure
    keras.backend.clear_session()

comptime = (time.time() - tic)/60
print("Finished\nCompletion Time (mins): " + str(comptime))

f = open(filename+"/generic_fitting_data.txt", "a")
f.write("Image Dimensions: (" + str(int(imagedimensions[0])) + ", " + str(int(imagedimensions[0])) + ")")
f.write("Overall Completion Time (mins): " + str(comptime))
f.close()
# save training data to neural network with a bunch of important parameters back to database
# use h5py