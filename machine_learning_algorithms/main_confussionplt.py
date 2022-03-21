##
# File for plotting the DNN confussion matrix from the same randomly selected simulation
# WILL PROBABLY NEED TO BE DONE ON THE SUPER COMPUTER DUE TO HARD DRIVE SIZE
#
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
from sklearn.metrics import plot_confusion_matrix,confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import DNN_run_mods
from tensorflow import keras
from joblib import load
from joblib import dump
import random

"""iMPORT DNN settings""" #Need to get which experimental parameters where loading
time1 = time.time()
input_name = sys.argv[1]

Numbersamples = 2000 # number of reaction to get from the sampled data


"""Load in a settings file with the location of the DNN models"""
DNNfile,clusterbayesloc = DNN_run_mods.DNN_model_locloader("DNNMODEL_LOCATION.txt")


serverdata, supivisorclass, ACCase, variate, DNNmodel, cpu_workers, gpu_workers, deci, harmdata, modelparamterfile, \
    trainratio, reactionmech = TDNN.inputloader(input_name)


# Something to classify the DNN label and data to a specific model
print(serverdata)
# THIS GETS THE LABELS

#if supivisorclass == "reactmech":  # classifiers the the DNN supivised to reaction mechs
    # """GET EXPsetrow THAT RELATE TO THE ABOVE AC SINE PROPERTIES AND  MODEL""" #exp = [[expsetrow,reactionmech],...]

exp_class = TDNN.EXPsetrow_collector(serverdata, reactionmech)

exp_data = TDNN.EXPsetrow_dcdata(serverdata, exp_class)

        # create list of form react_class = [[[ReactionID,Reactionmech]]] for each reac mech
react_class = TDNN.ReactID_collector(serverdata, exp_class)

#bunch of stuff to set it up for ratio and set it up to just test stuff
hold = []
for l in react_class:
    random.shuffle(l)
    h = []
    for i in range(Numbersamples):
        h.append(l[i])
    hold.append(h)

react_class = hold
del hold

trainratio = 1


"""BELOW NEEDS TO BE DONE TO take in a limit function OR A SEPERATE FUNCTION TO GENERATE"""
 # train data is for train NN, testdata is for testing NN
testdata, traindata, Ntest, Ntrain = TC_mod.suffle_splittrainratio(trainratio, react_class)

harmtrain, harmtrain_mech, harmtrain_ID = NNt.DC_NN_setter(traindata, harmdata)  # training data

# small change made fater less likely to parralellise it
currentdata, mechaccept = TDNN.ACsqlcurrentcollector(serverdata, harmdata, harmtrain, deci,DNNmodel)

# sets up model numbers FOR REACT MECH ONES
model_numcode = {}
for i in range(len(reactionmech)):
    model_numcode.update({reactionmech[i]:i})

#converts to a matrix as im lazy
print(harmtrain_mech)
print(model_numcode)
for i in range(len(harmtrain_mech)):
    harmtrain_mech[i] = model_numcode.get(harmtrain_mech[i])
print(harmtrain_mech)

Nmodels = len(reactionmech)

"""LOAD IN THE CLUSTERING FILE"""
if type(clusterbayesloc) != None:
    clusterbayes = DNN_run_mods.clusterbayesloader(clusterbayesloc)
    bayesmatrix = np.zeros((len(clusterbayes[0]),len(clusterbayes)))
    for i in range(len(clusterbayes)):
        j = 0
        modellabels = []  # too lazy to do it proper
        for key,items in clusterbayes[i].items():
            modellabels.append(key)
            bayesmatrix[j,i] = items
            j += 1

"""May need something here to import model specific parpameters"""
"""Load the reactmech or classes from the sql libary where exp setting is X"""




#make file to store confussion matrixes
filename = NN_train_mod.genericoutpufile(input_name+"CM")


"""WILL NEED SOMETHING IN THE ABOVE TO RANDOMLY SELECT WHAT ONES TO GET"""

""""""
"""RUNNING STUFF"""
""""""

"""SET UP A COPY OF THE DATA FOR ALL AVALIBLE MODES"""
#below been seperated into two for memory
modeltype = list(DNNfile.keys())
print(modeltype)
Rscurr = []
for Gencurrent in currentdata:
    dic = {}
    for i in range(len(Gencurrent)):
        dic.update({str(i): Gencurrent[i]})

    Rscurr.append(pd.DataFrame(dic))

#Rscurr = datahold
#print(Rscurr.shape)

datahold = []
for Gencurrent in currentdata:
    datahold.append(Gencurrent)

Iscurr =np.array(datahold)
r = Iscurr.shape
#Iscurr = Iscurr.transpose(0, 1, 2)
print(Iscurr.shape)

# again deletes stuff to free up memory
del currentdata
del datahold

"""LOAD THE DNN MODEL """
from sktime.transformations.panel.rocket import Rocket
for key, items in DNNfile.items():

    if key[0] == "I":
        model = keras.models.load_model(items)
        y_pred = model.predict(Iscurr)
        # load the inception time model

        # set input data to Rocket mode

    elif key[0] == "R":
        print(items)
        model = load(items, None) #might need a correction here for .joblib file
        # load the Rocket model
        kernelloc = items.split(".")
        kernelloc = kernelloc[0]+ "_kernel."+kernelloc[1]
        print(kernelloc)
        rocket = load(kernelloc,None) #  fix

        """NEED TO LOAD IN ROCKET AND CLASSIFIER"""
        y_pred = []
        #for values in Rscurr:
        #print(value)
        transformedcurr = rocket.transform(np.array(Rscurr))
        y_pred = model.predict(transformedcurr)
        #y_pred.append(y[0])
        # set input data to Rocket mode
        # do prediction
        """For testing remove later"""
        del model
        del rocket
    else:
        print("ERROR MODEL TYPE IS NOT SUPPORTED")
        exit()

    #print prediction
    if key[1] == "R": # reaction mech labels
        """DERP PRINT"""
        maxpred  = y_pred

    elif key[1] == "C": #clastering label reaction mechanisms
        r,c = bayesmatrix.shape
        contintable = np.zeros((r,c))
        bayetot = []
        """THIS IS GOING TO BE A NIGHTMARE"""
        if key[0] == "R":
            print(bayesmatrix)
            bayetothold = []
            print("FUCK IM SO TIRED")
            for k in range(len(y_pred)):
                print(y_pred[k],bayesmatrix[:, y_pred[k]])
                bayetothold.append(bayesmatrix[:, y_pred[k]])
            y_pred = bayetothold
            maxpred = []
            print(bayetothold)
            print("DDDDDDDDDDDDDDDDDDDDDDIIIIIIIIIIIIIIIIIIIIIICCCCCCCCCCCCCCKKKKKKKKKKKK")
            for values in bayetothold:
                maxpred.append(list(values).index(max(values)))  # list is like super lazy
            print(maxpred)
        elif key[0] == "I":
            print(bayesmatrix)
            bayetothold = []
            for k in range(len(y_pred)):
                bayetot = []
                for i in range(r):
                    for j in range(c):
                        contintable[i,j]= y_pred[k][j]*bayesmatrix[i,j]
                    bayetot.append(sum(contintable[i,:]))
                bayetothold.append(bayetot)
            y_pred = bayetothold

    # correction to get max value
    """FIX THIS SUNS RISING AND IM TIRED"""
    #print(y_pred)
    #print(key)
    #print(type(y_pred))
    if key[0] == "I":
        maxpred = []
        maxpred2nd = []
        for values in y_pred:
            nxm = list(values).index(max(values))
            maxpred.append(nxm)
            maxpred2nd.append(list(values).index(max(values[:nxm]+values[nxm+1:]))) # list is like super lazy


    #print(key, y_pred)

    models1 = modellabels #this labels the axis
    confussion = confusion_matrix(harmtrain_mech,maxpred,normalize="true")
    print(confussion)
    disp = ConfusionMatrixDisplay(confusion_matrix=confussion,display_labels=models1 )
    disp.plot()
    plt.savefig(filename+"/confussionmatrix_maxpredict"+str(key)+".png")
    plt.close()

    models1 = modellabels  # this labels the axis
    confussion = confusion_matrix(harmtrain_mech, maxpred2nd, normalize="true")
    print(confussion)
    disp = ConfusionMatrixDisplay(confusion_matrix=confussion, display_labels=models1)
    disp.plot()
    plt.savefig(filename + "/confussionmatrix_2ndmaxpredict" + str(key) + ".png")
    plt.close()

    # below are for the case if acceptance is below a number
    """models.append("None")
    #models[2] = "EC1st"
    print(models)
    confussion = confusion_matrix(true_ouputv,prediction_ouputv)
    print(confussion)
    disp = ConfusionMatrixDisplay(confusion_matrix=confussion,display_labels=models )
    disp.plot()
    plt.savefig(str(modeltype)+"confussionmatrix"+str(accept)+".png")
    plt.close()
    
    confussion = confusion_matrix(true_ouputv,prediction_ouputv,normalize="true")
    print(confussion)
    disp = ConfusionMatrixDisplay(confusion_matrix=confussion,display_labels=models )
    disp.plot()
    plt.savefig(str(modeltype)+"confussionmatrixnorm"+str(accept)+".png")
    """
    print("finished")
