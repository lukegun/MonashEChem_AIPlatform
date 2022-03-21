#placeholderpythonfile
import matplotlib.pyplot as plt
import pandas as pd

import DNN_run_mods
import DNNtimecluster_mod as TC_mod
import TrainDNNtimeclass_mod as TDNN
import DC_neuralnetwork_mod as NNt
import ImagebasedDNN as IB_NNt
import time
import sys
import numpy as np
import DNN_run_mods as runner_mod
import window_func as wint
from joblib import load
from scipy.fftpack import rfft
from tensorflow import keras
# python3.7 main_run.py AG4AC1.txt AG4AC8.txt AG4AC10.txt -dnn reactmech rocket -p D=6.5E-6cm2/s A=3.36e-02cm2 C=1.0mM k0=100cm/s
# where the thing after -dnn [[reactmech,clustering],[rocket,inceptiontime]]

print(sys.argv)

expfile,DNNarch,parameterestimates = runner_mod.terminalextract(sys.argv)

# extracts a constant dic from terminal line
paraestdic = runner_mod.paraextract(parameterestimates)

print(expfile)
print(DNNarch)
print(parameterestimates)


#"""Something to load in some settings"""
# these are currently placeholder
serverdata, supivisorclass, ACCase, variate, DNNmodel, cpu_workers, gpu_workers, deci, harmdata, modelparamterfile, \
    trainratio, reactionmech = TDNN.inputloader("test.txt")

""" Get experimental parameters """
Evaluehold = []
ACsettingshold = []
scanratehold = []
currenthold = []
voltagehold = []
for file in expfile:
    Evalues, ACsettings,scanrate,current,expvoltage,timedata = runner_mod.expextract(file)
    Evaluehold.append(Evalues)
    ACsettingshold.append(ACsettings)
    scanratehold.append(scanrate)
    currenthold.append(current)
    voltagehold.append(expvoltage)


# check to see if exp files are the same
if all([Evaluehold[0] == x1 for x1 in Evaluehold]) and all([ACsettingshold[0] == x1 for x1 in ACsettingshold]) and all([scanratehold[0] == x1 for x1 in scanratehold]):
    Evalues = Evaluehold[0]
    ACsettings = ACsettingshold[0]
    scanrate = scanratehold[0]
    # """Average the experimental voltage""" something can be added here for the psuedo reference
    expvoltage = np.average(voltagehold,axis= 0)

else:
    print("ERROR the exp settings files are wrong and don't all match up")
    exit()

""" identify if AC or DC """
ACmode = False
if ACsettings[1] != 0:
    ACmode = True

"""LOAD IN ALL THE DNN SPECIFIC PARAMETERS CAN USE THE TRAINING FILE FOR THIS"""
maxharmonicsinDNN = 9
DNNlength = 2**8
DNNdeci = int(len(expvoltage)/DNNlength)

""" extract the harmonics """
""" Treat experimental data into specific format """
if ACmode:

    std = 0.1
    AutoNharm = 0.5
    HarmMax = 12
    #print("pooo")
    #print(ACsettings)
    freqband = ACsettings[0]*3/5
    #print(freqband)
    bandwidth = []
    for j in range(2):
        band = []
        for i in range(12): # this needs to be 12
            band.append(freqband)
        bandwidth.append(band)

    # generate a bandwidth approx for 8 harmonics as a function of frequency
    nexpdeci,frequens = runner_mod.Simerharmtunc(len(currenthold[0]), timedata[0], bandwidth, 12, [ACsettings[0]])
    #print(nexpdeci)
    #Count number of harmonics
    Nharm, fft_res = runner_mod.harmoniccounter(currenthold[0], nexpdeci, AutoNharm)

    #this is here to check to see if Nharm is more then the allowed harmonics
    if maxharmonicsinDNN <= Nharm:
        Nharm = maxharmonicsinDNN

    bandwidthplot = np.array(bandwidth)
    bandwidth = bandwidthplot[:,:Nharm]

    # generate filters
    #simfilterhold = wint.RGguassconv_filters(2**Nsim, bandwidth, timedata[1]/(2**Nsim), [ACsettings[0]], std)
    expfilterhold = wint.RGguassconv_filters(len(currenthold[0]), bandwidth, timedata[0], [ACsettings[0]], std)

    # generate all harmonics
    hil_storehold = []
    fundharmhold = []
    for current in currenthold:
        fft_res = rfft(current)
        hil_store,fundimentalharm = runner_mod.Harmonic_gen(fft_res, expfilterhold)
        hil_storehold.append(hil_store)
        fundharmhold.append(fundimentalharm)

    #average the fundimental for capacitance stuff
    fundimentalharm = np.average(fundharmhold, axis=0)
    del fundharmhold

    # average each harmonic
    xy = []
    for i in range(Nharm + 1):
        harms = []
        for hils in hil_storehold:
            harms.append(hils[i][:])

        harm = np.average(np.array(harms), axis=0)
        xy.append(harm)
    hilstore = xy

    percentagetrun = 0.05
    truntime = percentagetrun*timedata[1]

    trunvec_sim, trunvec_exp = runner_mod.EXPharmtreatment(len(hilstore[0]), timedata[1], [truntime,  timedata[1] - truntime ], 2**16)
    del trunvec_sim

    #stored for the capacitance fitting before truncatio
    capcurrent = hilstore[1]
    xy = []
    for hils in hil_store:
        harm = wint.Average_noiseflattern(hils, trunvec_exp, avgnum=20)
        xy.append(harm[::DNNdeci]) # adds decimation in

    current = xy
    "truncate edges"

    del  hil_storehold

else: # DC mode
    current = np.average(currenthold,axis= 0)

    # stored for the capacitance fitting before truncation
    capcurrent = current[::DNNdeci] #add the decimation in
    del currenthold # clear up ram
    fundimentalharm = None # required due to ac process
    ACsettings = None


"""decimate the time series to the setting file"""

""" NON_dimensionalise the current """ #need to consider the Non dim and effect of surface and diff difference and the effect on this
print("Below normalisation doesn't work right remove")
scurr, constant = runner_mod.currentnondim(currenthold[0],paraestdic.get(6), paraestdic.get(22), paraestdic.get(21))
#Gencurrent = scurr
#constant = 1 #for testing
Gencurrent = runner_mod.normalisationrange(current,constant)

# will need a correction for when this isn't the case
temp = 273.15 + 23
surfaceconfinedcurr, constant = runner_mod.currentnondimsurfconfined(currenthold[0],paraestdic.get(6), paraestdic.get(21), scanrate, temp)
#constant = 1 #for testing
surfcurrent = runner_mod.normalisationrange(current,constant)
#surfcurrent = [scurr*constant/max(scurr) for scurr in current]

#adds the empty array in so no divide by zero issue
if Nharm < maxharmonicsinDNN and ACmode:
    for i in range(len(hilstore), maxharmonicsinDNN):  # this adds the blank harmonics if none where found
        Gencurrent.append(np.zeros(DNNlength))
        surfcurrent.append(np.zeros(DNNlength))

if ACmode: # AC mode
    pass
else: # DC case
    pass

"""Load in a settings file with the location of the DNN models"""
DNNfile,clusterbayesloc = DNN_run_mods.DNN_model_locloader("DNNMODEL_LOCATION.txt")


"""LOAD IN THE CLUSTERING FILE"""
if type(clusterbayesloc) != None:
    clusterbayes = DNN_run_mods.clusterbayesloader(clusterbayesloc)
    print(clusterbayes)
    bayesmatrix = np.zeros((len(clusterbayes[0]),len(clusterbayes)))
    for i in range(len(clusterbayes)):
        j = 0
        for key,items in clusterbayes[i].items():
            bayesmatrix[j,i] = items
            j += 1

"""SET UP A COPY OF THE DATA FOR ALL AVALIBLE MODES"""
modeltype = list(DNNfile.keys())
print(modeltype)
if any("R"==i[0] for i in modeltype): # rocket classifiers are used
    dic = {}
    for i in range(len(Gencurrent)):
        dic.update({str(i): Gencurrent[i]})

    print(len(Gencurrent))
    Rscurr = pd.DataFrame([dic])
    print(Rscurr.shape)
if any("I"==i[0] for i in modeltype): # inception time classifiers are used
    print(type(Gencurrent))
    Iscurr =np.array([Gencurrent])
    r = Iscurr.shape
    Iscurr = Iscurr.transpose(0, 2, 1)

for i in range(r[1]):
    plt.figure()
    plt.plot(Iscurr[0,:,i])
    plt.savefig(str(i)+".png")
    plt.close()

"""LOAD THE DNN MODEL """
from sktime.transformations.panel.rocket import Rocket
print(DNNfile)
DNNfile = {'IC': 'DNN_MODELS/inceptiontime_clustering.hdf5'}#
for key, items in DNNfile.items():

    if key[0] == "I":
        model = keras.models.load_model(items)
        y_pred = model.predict(Iscurr)
        # load the inception time model

        # set input data to Rocket mode

    elif key[0] == "R":
        model = load(items, None) #might need a correction here for .joblib file
        # load the Rocket model
        kernelloc = items.split(".")
        kernelloc = kernelloc[0]+ "_kernel."+kernelloc[1]
        print(kernelloc)
        rocket = load(kernelloc,None) #  fix

        """NEED TO LOAD IN ROCKET AND CLASSIFIER"""
        transformedcurr = rocket.transform(Rscurr)
        y_pred = model.predict(transformedcurr)
        print("shit")
        print(y_pred)
        # set input data to Rocket mode
        # do prediction
        """For testing remove later"""


    else:
        print("ERROR MODEL TYPE IS NOT SUPPORTED")
        exit()

    #print prediction
    if key[1] == "R": # reaction mech labels
        """DERP PRINT"""
        y_predf  = y_pred

    elif key[1] == "C": #clastering label reaction mechanisms
        r,c = bayesmatrix.shape
        contintable = np.zeros((r,c))
        bayetot = []
        """THIS IS GOING TO BE A NIGHTMARE"""
        if key[0] == "R":
            print(bayesmatrix)
            bayetothold = []
            print(y_pred[0], bayesmatrix[:, y_pred[0]])
            bayetothold.append(bayesmatrix[:, y_pred[0]])
            y_pred = bayetothold
            maxpred = []
            print(bayetothold)
        elif key[0] == "I":
            print(bayesmatrix)
            for i in range(r):
                for j in range(c):
                    contintable[i,j]= y_pred[0][j]*bayesmatrix[i,j]
                bayetot.append(sum(contintable[i,:]))
        print(contintable)
        print(bayetot)


    print(key, y_pred)

"""GET THE PRIOR DATA IF INCEPTION TIME """
#load bayes file
# get 2 most likely reactions and inception label

# get a parameter distabution of reaction mechanims with the label

# will also need to fix this all up for use on runner in a GCP setting

""" PRINT THE OUTPUT for a person"""

"""Print something to pass to the greedy output"""
