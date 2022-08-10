import numpy
import matplotlib.pyplot as plt
import time
from copy import deepcopy
import datetime
import os
import numpy as np
import pandas as pd
from tslearn.clustering import KShape, KernelKMeans,TimeSeriesKMeans
from tslearn.neighbors import KNeighborsTimeSeries
from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
import cluster_mod as Cmod
import sys
import shutil
from tslearn.clustering import TimeSeriesKMeans

"""tslearn requires numpy 1.20 whereas tensorflow uses 1.19"""

#issue in dtw for the tslearn package creates a massive matrix and wrecks my RAM check before use
#l1 = 15000
#l2 = 15000
#cum_sum = numpy.full((l1 + 1, l2 + 1), numpy.inf)
#print(cum_sum.nbytes/10**6)
#exit()
time1 = time.time()
input_name = sys.argv[1]
# import loader input file
serverdata, filpcurrent, fliprate, AC_case, cpu_workers, deci, nondim, reactionmech, n_cluster, harmoicnumber, totn_cluster, n_init, metric,training = Cmod.inputloader(input_name)

#metric : {“euclidean”, “dtw”, “softdtw”
"""Get reaction ID being used"""
reactID = Cmod.sqlReacIDPreallo(serverdata,reactionmech)

# sets up output file
x = input_name.split(".")
input_name0= x[0]
s = x[0]

filename = Cmod.outputfilegenertor(input_name0)

outputname = filename+"/" + "Bayesoutput.txt"
f = open(outputname,"w+")
f.close()

shutil.copyfile(input_name, filename+"/settings.txt")

if AC_case:
    print("AC")

    Cmod.ACLabelpreallocator(serverdata,reactID)
    print("allocated")

    i = 0
    ogmetric = deepcopy(metric)
    for harmnum in harmoicnumber:
        n_cluster = totn_cluster[i]
        print(harmnum)
        itertime = time.time()

        # truncation for the lower harmonics where higher accuracy isnt required
        # with this lowering the amount of RAM required for the cases where theres a lot of groups
        if harmnum == 0 or harmnum == 1 or harmnum == 2 or harmnum == 3:
            ftacvdeci = int(deci/2)
            if metric[1] != 0:
                metric[1] = int(metric[1]/2)
        else:
            ftacvdeci = deci
        t1 = time.time()
        currentdata, mechaccept = Cmod.sqlcurrentcollector(serverdata, harmnum, reactID, ftacvdeci)
        print("Data collection time: " + str((time.time() - t1) / 60))
        #Cmod.svm(currentdata,mechaccept,reactionmech,cpu_workers)
        #exit()

        #Cmod.svm(currentdata,mechaccept,reactionmech,cpu_workers)
        print("working")
        print(training)
        t1 = time.time()
        if training[0]:
            y_pred = Cmod.TseriesKmeans(currentdata,n_cluster,cpu_workers,mechaccept,reactionmech,harmnum,n_init,filename,metric)
        else: # loads fitted model from file
            print("lalala")
            y_pred = Cmod.TseriesKmeansloader(currentdata,harmnum,training[1])
        metric = ogmetric
        print("Prediction time: " + str((time.time()-t1)/60))


        strlabel = Cmod.harmlabeltochar(mechaccept,y_pred,harmnum)
        #"""Load each time series data into an array, use a list of np.array"""
        """FIX FOR SPEED"""
        t1 = time.time()
        Cmod.harmlabel(serverdata,strlabel,harmnum)
        print("Label Input time: " + str((time.time() - t1)/60))
        """load corriponding y-values"""
        #dists,ind = Cmod.KnearestN(currentdata,n_clusters,cpu_workers)
        comtime = (time.time()- itertime)/60

        Cmod.outputwritter(outputname, comtime, harmnum, y_pred, reactionmech, n_cluster, mechaccept)

        i += 1

    """FIX FOR SPEED"""
    t1 = time.time()
    labelocurrannce = Cmod.AClabelcollector(serverdata, reactID, totn_cluster,harmoicnumber)
    print("OVERALL final LABELING TIME: " + str((time.time() - t1)/60))

    comtime = (time.time() - time1) / 60
    Cmod.outputwritterfin(outputname, comtime, labelocurrannce)


else:
    print("DC")
    itertime = time.time()
    harmnum = -1
    n_cluster = totn_cluster[0]

    currentdata, mechaccept = Cmod.sqlcurrentcollector(serverdata, harmnum, reactID, deci)

    # Cmod.svm(currentdata,mechaccept,reactionmech,cpu_workers)
    # exit()

    # Cmod.svm(currentdata,mechaccept,reactionmech,cpu_workers)
    kmeantime  = time.time()
    if training[0]:
        y_pred = Cmod.TseriesKmeans(currentdata, n_cluster, cpu_workers, mechaccept, reactionmech, harmnum,n_init,filename,metric)
    else: # loads fitted model from file
        y_pred = Cmod.TseriesKmeansloader(currentdata, harmnum, training[1])

    kmeantime = (time.time() - kmeantime)/60
    print(metric[0]+" time (mins) " +str(kmeantime))
    strlabel = Cmod.harmlabeltochar(mechaccept, y_pred, harmnum)
    # """Load each time series data into an array, use a list of np.array"""
    #Cmod.harmlabel(serverdata, strlabel, harmnum)

    #something to print string labels to a .txt
    Cmod.labelprinterDC(filename,strlabel)

    """load corriponding y-values"""
    # dists,ind = Cmod.KnearestN(currentdata,n_clusters,cpu_workers)
    comtime = (time.time() - itertime) / 60

    Cmod.outputwritter(outputname, comtime, harmnum, y_pred, reactionmech, n_cluster, mechaccept)

    labelocurrannce = Cmod.DClabelcollector(serverdata,strlabel,[n_cluster])

    comtime = (time.time() - time1) / 60
    Cmod.outputwritterfin(outputname,comtime,labelocurrannce)
    """print label occurance to file"""





print("finished")
exit()
