import scipy as sci
from scipy.special import jv, yn    # jv = first order, yn =second kind
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import time
import string
import psycopg2
from tensorflow import keras
import TrainDNNtimeclass_mod as TDNN
import DNN_run_mods as runner_mod
from classifiers.inception import Classifier_INCEPTION

#This is the cluster
modelfile = "DNN_MODELS/inceptiontime_clustering.hdf5"
value = 5 # number of clusters

fogsize = (16,8)
fontsize = 15

model = keras.models.load_model(modelfile)
#barycenter = model.cluster_centers_[modellabel].ravel()

dname = ["postgres", "Quantum2", "35.193.220.74", 5432, "ftacv-test"]
DNNmodel = "inceptiontime"

arrs = []
count = 0
arrlabels = []

deci = 2**8

"""LOAD THE REACTION MECHANISMS data with resistance, KINETICS and REACTION ID"""
reactmech = ["E","EC","EE"]
harmdata = [0,1,2,3,4,5,6,7,8]
"""LOOP FOR A BUNCH OF REACTION MECHANISMS"""
for rm in reactmech:

    try:
        connection = psycopg2.connect(user=dname[0],
                                      password=dname[1],
                                      host=dname[2],
                                      port=dname[3],
                                      database=dname[4])

        cursor = connection.cursor()
        para = cursor.execute(
            """SELECT "Reaction_ID", "ksreal","Ru" FROM "Simulatedparameters" WHERE "ReactionMech" = %s""", (rm,))
        para = []
        for table in cursor.fetchall():
            para.append(table)

        # sqlcom = """INSERT INTO time(test,"Ctime") VALUES( %()s, CURRENT_TIMESTAMP )"""

    except (Exception, psycopg2.Error) as error:
        print("error,", error)
    finally:
        # This is needed for the
        if (connection):
            cursor.close()
            connection.close()

    """TRUNCATE THE REACTION ID FOR TESTING"""
    para = para[:2000]

    """CONVERT THE REACTION_ID  into a tuple and pair ko and ru with ID into a dic"""
    paradic = {}
    reacttuple = []
    for values in para:
        reacttuple.append([values[0], rm])
        if rm == "EE":
            k0 = (values[1][0] + values[1][1])/2
        else:
            k0 = values[1][0]
        # add the values to a dic
        paradic.update({values[0]:[k0,values[2]]})
    #reacttuple = tuple(reacttuple)


    """USE REACTION ID TO GET THE HARMONIC INFORMATION"""
    currentdata, mechaccept = TDNN.ACsqlcurrentcollector(dname, harmdata, reacttuple, deci,DNNmodel)

    """NORMALIZE THE HARMONICS"""
    """constant = 1
    currenttot = []
    for scurr in currentdata:
        datatot = runner_mod.normalisationrange(scurr, constant)
        currenttot.append(datatot)
    currentdata = currenttot
    del datatot
    del currenttot"""


    """CONVERT INTO THE right format"""
    #if any("I" == i[0] for i in modeltype):  # inception time classifiers are used
    #print(type(Gencurrent))
    Iscurr = np.array(currentdata)
    #r = Iscurr.shape
    #Iscurr = Iscurr.transpose(0, 2, 1)

    """TEST AND PREDICT THE SIMULATIONS """
    y_pred = model.predict(Iscurr)
    #print(y_pred)
    """This is legitimatly converts the max value to an index"""
    indexpred = []
    for stuff in y_pred:
        j = np.where(stuff == np.amax(stuff))
        indexpred.append(j[0][0])

    ptot = []
    for i in range(len(stuff)):
        p = []
        for j in range(len(indexpred)):
            if indexpred[j] == i:
                p.append([paradic.get(mechaccept[j][0])[0],paradic.get(mechaccept[j][0])[1]]) #k0, Ru
        print(len(p))
        ptot.append(p)





    """GETS ARRAY OF THE INDEX INPUTS"""

    """PLOT THE STUFF OUT"""
    fig, axs = plt.subplots(1, 2, tight_layout=True, figsize=fogsize)
    plotnum = []
    i = 0
    #print(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    clo = plt.rcParams['axes.prop_cycle'].by_key()['color']
    #print(clo)
    x = []
    colour = [clo[0], clo[1], clo[3], clo[2]]
    ii = 0
    for l in ptot:

        if len(l) >= 1:#100:
            l = np.array(l)
            #print(l)
            print(l.shape)
            axs[0].scatter(l[:, 0], l[:, 1], alpha=0.5) #, color=colour[ii]
            axs[0].set_xscale("log")
            axs[0].set_yscale("log")
            axs[1].hist(np.log10(l[:, 1]), 50, alpha=0.5)
#            plotnum.append(l[np.random.randint(l.shape[0]), 2])  # gets a random thing to plot
            # axs[1].set_xscale("log")
            x.append(len(l[:, 0]))
            ii += 1
        i += 1

    #axs[2].legend(x, loc='lower right', title="Number of\nsimulations", fontsize=fontsize, title_fontsize=fontsize)
    axs[0].set_xlabel("$k^0$", fontsize=fontsize)
    axs[0].set_ylabel("$R_u$", fontsize=fontsize)
    axs[1].set_xlabel("$Log_{10}(R_u)$", fontsize=fontsize)
    axs[1].set_ylabel("Frequency", fontsize=fontsize)


    fig.suptitle(rm, fontsize=fontsize * 2)
    fig.savefig(rm + "atparas.png")
    plt.close()




