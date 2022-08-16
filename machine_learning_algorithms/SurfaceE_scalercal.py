#This is the code to calculate the scalar value of the surface confined to E mechanism then print it
# The value is caluclated from the ratio of the max first harmonic
import matplotlib.pyplot as plt
import pandas as pd

import DNNtimecluster_mod as TC_mod
import NN_train_mod
import TrainDNNtimeclass_mod as TDNN
import DC_neuralnetwork_mod as NNt
import ImagebasedDNN as IB_NNt
import time
import sys
import numpy as np
import shutil
import DNN_run_mods as runner_mod
from joblib import dump
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=1424)])


"""iMPORT DNN settings""" #Need to get which experimental parameters where loading
time1 = time.time()

#This is the output of input loader but is constant in this calculation
serverdata = ["postgres", "password", "ipadress", "post", "table"]
reactionmech = ["E","ESurf"]
modelparamterfile = "testmodelfile.txt" # don't know what this does but leave it in and hope it works
deci = 2**8
DNNmodel = "inceptiontime"

# Something to classify the DNN label and data to a specific model
print(serverdata)
# THIS GETS THE LABELS


# """GET EXPsetrow THAT RELATE TO THE ABOVE AC SINE PROPERTIES AND  MODEL""" #exp = [[expsetrow,reactionmech],...]

exp_class = TDNN.EXPsetrow_collector(serverdata, reactionmech)
exp_data = TDNN.EXPsetrow_dcdata(serverdata, exp_class)

# create list of form react_class = [[[ReactionID,Reactionmech]]] for each reac mech
react_class = TDNN.ReactID_collector(serverdata, exp_class)
#print(react_class)

# count number for each reaction model and check if same
Narray = TDNN.Narray_count(react_class)


modeldic = TDNN.modeldataloader(modelparamterfile)
"""May need something here to import model specific parpameters"""
"""Load the reactmech or classes from the sql libary where exp setting is X"""

# sets up model numbers
model_numcode = {}
for i in range(len(reactionmech)):
    model_numcode.update({reactionmech[i]:i})
Nmodels = len(reactionmech)

# tell everything just to get the fundimental harmonic
harmdata = [1]
#print("Train data")
#print(traindata)

# extract the E stuff
react_classE = [react_class[0]]
testdataE1, traindata, Ntest, Ntrain = TC_mod.suffle_splittrainratio(0.1, react_classE)
harmE1, harmE1_mech, harmE1_ID = NNt.DC_NN_setter(traindata, harmdata)  # training data
currentdataE1, mechacceptE1 = TDNN.ACsqlcurrentcollector(serverdata, harmdata, harmE1, deci,"blah")
NE1 = len(currentdataE1)


# extract the Esurf stuff
react_classEsurf = [react_class[1]]
testdataEsurf , traindataEs, Ntest, Ntrain = TC_mod.suffle_splittrainratio(0.99, react_classEsurf)
harmEsurf, harmEsurf_mech, harmEsurf_ID = NNt.DC_NN_setter(traindataEs, harmdata)  # training data
currentdataEsurf, mechacceptEsurf = TDNN.ACsqlcurrentcollector(serverdata, harmdata, harmEsurf, deci,"blah")
NEsurf = len(currentdataEsurf)

import matplotlib.pyplot as plt
plt.figure()
plt.plot(currentdataEsurf[0]*7638918.383354617)
plt.plot(currentdataE1[0])
plt.savefig("savefig.png")

# Something to randomly compare the
xdiff = []
t2 = time.time()
Nr = 300000
for i in range(Nr):
    x = np.random.randint(0,NEsurf-1,2)
    xdiff.append(max(currentdataE1[x[0]])/max(currentdataEsurf[x[1]]))

#print(xdiff)
print("finished")
print((time.time()-t2))
print(len(xdiff))
print(np.average(xdiff))
