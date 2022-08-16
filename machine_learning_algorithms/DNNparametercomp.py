import matplotlib.pyplot as plt
import numpy as np
import time
import string
import psycopg2
from tensorflow import keras
import tensorflow as tf
import TrainDNNtimeclass_mod as TDNN
import DNN_run_mods as runner_mod
from classifiers.inception import Classifier_INCEPTION

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=1024)])

#This is the cluster
value = 8 # number of clusters

fogsize = (16,8)
fontsize = 15

#barycenter = model.cluster_centers_[modellabel].ravel()
ACcase = True

if ACcase:
    modelfile = "DNN_MODELS/AC/inceptiontime_clustering.hdf5"
    dname = ["postgres", "password", "host", post, "table"]
    harmdata = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    accase = "AC" # for naming
else:
    modelfile = "DNN_MODELS/DC/inceptiontime_clustering.hdf5"
    dname = ["postgres", "password", "host", port, "table"]
    harmdata = [-1]
    accase = "DC" # for naming

model = keras.models.load_model(modelfile)

DNNmodel = "inceptiontime"

arrs = []
count = 0
arrlabels = []

deci = 2**8

"""LOAD THE REACTION MECHANISMS data with resistance, KINETICS and REACTION ID"""
reactmech = ["EE","ESurf","ECE","E","ECat","EC"]
reactmech = ["EC"]

"""LOOP FOR A BUNCH OF REACTION MECHANISMS"""
for rm in reactmech:

    try:
        connection = psycopg2.connect(user=dname[0],
                                      password=dname[1],
                                      host=dname[2],
                                      port=dname[3],
                                      database=dname[4])

        cursor = connection.cursor()
        if  rm == "EC":
            para = cursor.execute(
                """SELECT "Reaction_ID",  "Ru","kforward" FROM "Simulatedparameters" WHERE "ReactionMech" = %s""", (rm,))
            #"""SELECT "Reaction_ID",  "ksreal","Ru" FROM "Simulatedparameters" WHERE "ReactionMech" = %s""", (rm,))
            # """SELECT "Reaction_ID",  "Ru","kforward" FROM "Simulatedparameters" WHERE "ReactionMech" = %s""", (rm,))
            yy = "k_f"#yy = "R_u"
            xx="$R_u$"#xx = "$k^0$"
            xscale = "log"
            yscale = "log"
        elif rm == "ECat":
            para = cursor.execute(
                """SELECT "Reaction_ID", "Ru", "kforward" FROM "Simulatedparameters" WHERE "ReactionMech" = %s""",
                (rm,))
            yy = "k_f"
            xx = "$R_u$"
            xscale = "log"
            yscale = "log"

        elif rm == "EE" or rm == "ECE":
            para = cursor.execute(
                """SELECT "Reaction_ID", "Ru", "formalE" FROM "Simulatedparameters" WHERE "ReactionMech" = %s""",
                (rm,))
            yy = "\Delta E^0"
            xx = "$R_u$"
            xscale = "linear"
            yscale = "log"

        elif rm == "ESurf":
            para = cursor.execute(
                """SELECT "Reaction_ID", "ksreal","Ru" FROM "Simulatedparameters" WHERE "ReactionMech" = %s""", (rm,))
            xx = "$k^0$"
            yy = "R_u"
            xscale = "log"
            yscale = "log"

        else:
            para = cursor.execute(
            """SELECT "Reaction_ID", "Ru","ksreal" FROM "Simulatedparameters" WHERE "ReactionMech" = %s""", (rm,))
            xx = "$k^0$"
            yy = "R_u"
            xscale = "log"
            yscale = "log"

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
    para = para[:4000] #2000

    """CONVERT THE REACTION_ID  into a tuple and pair ko and ru with ID into a dic"""
    paradic = {}
    reacttuple = []
    for values in para:
        reacttuple.append([values[0], rm,rm])
        if rm == "EE":
            ru = values[1]
            de = values[2][1] - values[2][0]
            paradic.update({values[0]: [de, ru]})

        elif rm == "ECE":
            kf = values[1]
            de = values[2][2] - values[2][0]
            paradic.update({values[0]: [de, kf]})

        elif rm == "ESurf":
            kf = values[2]
            k0 = values[1][0]
            paradic.update({values[0]: [kf, k0]})

        elif rm == "EC":
            kf = values[1]#kf = values[1][0]#
            k0 = values[2][1]#k0 = values[2]#
            paradic.update({values[0]: [k0, kf]})

        elif rm == "ECat":
            kf = values[2][1]
            k0 = values[1]
            paradic.update({values[0]: [kf, k0]})

        else:
            k0 = values[2][0]
            # add the values to a dic
            paradic.update({values[0]: [values[1], k0]})

    #reacttuple = tuple(reacttuple)


    """USE REACTION ID TO GET THE HARMONIC INFORMATION"""
    currentdata, mechaccept = TDNN.ACsqlcurrentcollector(dname, harmdata, reacttuple, deci,DNNmodel)

    # This is to do the corrections on surface concentration and to make the normalisation the same
    if True: # shouldn't be happening as there is no ESurfs now
        scalar = 7568546.0
        for i in range(len(mechaccept)):
            if mechaccept[i][1] == "ESurf":
                currentdata[i] = currentdata[i] * scalar

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
    i = 1
    #print(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    clo = plt.rcParams['axes.prop_cycle'].by_key()['color']
    cor = 25
    #print(clo)
    x = []
    clo.append('m')
    ii = 0
    row = []
    j = 0 # counter
    for l in ptot:
        toys = []
        if len(l) >= 100:#100:


            j += 1
            l = np.array(l)
            #print(l)
            print(l.shape)
            axs[0].scatter(l[:, 0], l[:, 1], alpha=0.5, color=clo[-i]) #, color=colour[ii]
            axs[0].set_yscale(yscale)
            axs[0].set_xscale(xscale)
            if xscale == "log":
                axs[1].hist(np.log10(l[:, 0]), 50, alpha=0.5, color=clo[-i])
            else:
                axs[1].hist(l[:, 0], 50, alpha=0.5, color=clo[-i])
#            plotnum.append(l[np.random.randint(l.shape[0]), 2])  # gets a random thing to plot
            # axs[1].set_xscale("log")
            x.append(len(l[:, 0]))
            ii += 1
            row.append(str(i))
        i += 1

    #axs[2].legend(x, loc='lower right', title="Number of\nsimulations", fontsize=fontsize, title_fontsize=fontsize)
    axs[0].set_ylabel(xx, fontsize=fontsize+cor)
    axs[0].set_xlabel("$"+yy+"$", fontsize=fontsize+cor)
    s = "$Log_{10}(%s)$" %yy
    axs[1].set_xlabel(s, fontsize=fontsize+cor)
    axs[1].set_ylabel("Frequency", fontsize=fontsize+cor)
    axs[1].legend(row,title = "Cluster\nNumber")

    if rm == "ECat":
        nreactttle = "$E_{Cat}$"
    elif rm == "ESurf":
        nreactttle = "$E_{Surf}$"
    else:
        nreactttle = "$"+rm +"$"

    fig.suptitle(nreactttle, fontsize=fontsize * 2 +cor)
    fig.savefig(rm +accase+ "atparas.png")

    #This is for standardisation
    xmin, xmax = axs[0].get_xlim()
    custom_xlim = (xmin,xmax)
    ymin, ymax = axs[0].get_ylim()
    custom_ylim = (ymin,ymax)

    plt.close()

    # this is for the supporting information
    bla = int(np.ceil(j/3))

    fogsize2 = (int(8*3), bla*8)
    print(fogsize)
    fig, axs = plt.subplots(bla, 3, tight_layout=True, figsize=fogsize2)
    j = 0
    i = 1
    jj = 0
    for l in ptot:

        if j == 3: # reset row
            j = 0
            jj += 1

        if len(l) >= 100:  # 100:
            l = np.array(l)
            # print(l)
            print(l.shape,i)
            print(clo)
            print(jj,j,i)
            if bla != 1:
                axs[jj,j].scatter(l[:, 0], l[:, 1], alpha=1, color=clo[-i])  # , color=colour[ii]
                axs[jj,j].set_yscale(yscale)
                axs[jj,j].set_xscale(xscale)
                axs[jj,j].set_ylabel(xx, fontsize=2*fontsize)
                axs[jj,j].set_xlabel("$" + yy + "$", fontsize=2*fontsize)
                s = "$C_{%i}$" %(i)
                axs[jj,j].set_title(s, fontsize = 2*fontsize)
            else:
                axs[j].scatter(l[:, 0], l[:, 1], alpha=1, color=clo[-i])  # , color=colour[ii]
                axs[j].set_yscale(yscale)
                axs[j].set_xscale(xscale)
                axs[j].set_ylabel(xx, fontsize=2 * fontsize)
                axs[j].set_xlabel("$" + yy + "$", fontsize=2 * fontsize)
                s = "$C_{%i}$" % (i)
                axs[j].set_title(s, fontsize=2 * fontsize)
            plt.setp(axs, xlim=custom_xlim, ylim=custom_ylim)

            #            plotnum.append(l[np.random.randint(l.shape[0]), 2])  # gets a random thing to plot
            # axs[1].set_xscale("log")
            x.append(len(l[:, 0]))
            ii += 1
            row.append(str(i))
            j += 1
        i += 1

    fig.suptitle(nreactttle, fontsize=fontsize * 3)
    fig.savefig("SI_"+rm + accase + "atparas.png")
    plt.close()




