import numpy as np
import psycopg2
import psycopg2.extras as extras
import time
import datetime
import matplotlib.pyplot as plt
import os
import random
from tslearn.preprocessing import TimeSeriesScalerMinMax, TimeSeriesScalerMeanVariance
from tslearn.clustering import KShape, KernelKMeans,TimeSeriesKMeans
from tslearn.neighbors import KNeighborsTimeSeriesClassifier, KNeighborsTimeSeries
from tslearn.svm import TimeSeriesSVC

def inputloader(input_name):

    with open(input_name) as f:
        lines = f.readlines()

    #serverdata in format ["serverdata"]

    serverdata = inputstrip(lines[0])
    serverdata = serverdata.split(",")
    x = []
    # order is [user, password, host, port, database]
    for values in serverdata:
        x.append(values.strip(" "))
    serverdata = x

    x = inputstrip(lines[1])
    filpcurrent = [x == "True" or x == "true"]
    filpcurrent = filpcurrent[0]

    fliprate = float(inputstrip(lines[2]))

    x = inputstrip(lines[3])
    AC_case = [x == "True" or x == "true"]
    AC_case = AC_case[0]

    cpu_workers = int(inputstrip(lines[4]))

    deci = 2**int(inputstrip(lines[5]))

    x = inputstrip(lines[6])
    nondim = [x == "True" or x == "true"]
    nondim = nondim[0]

    harmoicnumber = inputstrip(lines[7])
    harmoicnumber = harmoicnumber.split(",")
    x = []
    # order is [user, password, host, port, database]
    for values in harmoicnumber:
        x.append(int(values.strip(" ")))
    harmoicnumber = x

    n_cluster = inputstrip(lines[8])
    n_cluster = n_cluster.split(",")
    x = []
    # order is [user, password, host, port, database]
    for values in n_cluster:
        x.append(int(values.strip(" ")))
    n_cluster = x

    n_init = int(inputstrip(lines[9]))

    metric = inputstrip(lines[10])
    metric = metric.split(",")
    x = [str(metric[0].strip(" ")),int(metric[1].strip(" "))]
    # order is [metric,SC_band]
    metric = x

    # loaad up the process
    training = inputstrip(lines[11])

    training = training.split(",")
    x = [str_to_bool(training[0].strip(" ")), str(training[1].strip(" "))]
    # order is [metric,SC_band]
    training = x


    i = 9
    for line in lines[i:]:
        if line.strip("\n\t ") == "Reaction mechanisms":
            jj = i
            break
        i += 1

    reactionmech = []
    j = jj +1
    jj = j
    for line in lines[j:]:
        if line != "\n":
            reactionmech.append(inputstrip(line))
        else:
            break
        jj += 1


    return serverdata, filpcurrent, fliprate,AC_case,cpu_workers,deci,nondim, reactionmech, n_cluster, harmoicnumber,n_cluster, n_init,metric,training


def str_to_bool(s):
    if s == 'True':
         return True
    elif s == 'False':
         return False
    else:
         raise ValueError

def inputstrip(x):
    y = x.strip("\n\t ")
    y = y.split("#")[0]
    y = y.strip("\n\t ")
    y = y.split("=")[-1]
    y = y.strip(" \t")
    return y

# this is for saving the models too
def outputfilegenertor(outputfname):

    file = True
    i = 0
    while file:
        try:
            if i == 0:
                filename = outputfname
                os.makedirs(filename)
            else:
                filename = outputfname +"_V" +str(i)
                os.makedirs(filename)
            file = False
        except:
            print("file already exists")
        finally:
            i += 1

    return filename

# Function for preallocation of the Rection ID for pooling
def sqlReacIDPreallo(serverdata,reactmech):

    try:
        connection = psycopg2.connect(user=serverdata[0],
                                      password=serverdata[1],
                                      host=serverdata[2],
                                      port=serverdata[3],
                                      database=serverdata[4])

        cursor = connection.cursor()

        tots = []
        for mech in reactmech:
            cursor.execute("""SELECT "Reaction_ID" FROM "Simulatedparameters" WHERE "ReactionMech" = %s """, (mech,))
            xtot = cursor.fetchall()
            for x in xtot:
                tots.append([x[0],mech])

        random.shuffle(tots)
        random.shuffle(tots)

    except (Exception, psycopg2.Error) as error:
        print("error,", error)
    finally:
        # This is needed for the
        if (connection):
            cursor.close()
            connection.close()

    # insert the values into sql
    return tots

def sqlcurrentcollector(serverdata,current, reactionID,deci):

    try:
        connection = psycopg2.connect(user=serverdata[0],
                                      password=serverdata[1],
                                      host=serverdata[2],
                                      port=serverdata[3],
                                      database=serverdata[4])

        cursor = connection.cursor()

        yaya = sqlharmcol(current)

        tupleID = []
        for ID in reactionID:
            tupleID.append(ID[0])
        tupleID = tuple(tupleID)

        cursor.execute(yaya, (tupleID,))
        xtotall = cursor.fetchall()

        tots = []
        mechaccept = []
        for xrow in xtotall:
            xtot = xrow[0]
            reactID = xrow[1]
            #cursor.execute(yaya, (ID[0],))

            #xtot =cursor.fetchall()[0][0]

            if type(xtot) is type(None):

                pass
            else:
                xtot = np.array(xtot)
                nlen = len(xtot)
                Ndec = int(nlen / deci)
                xtot = xtot[::Ndec]

                xtot = xtot / max(xtot)
                tots.append(xtot)

                #slow but lazy and works (world need to change a lot of stuff to dics to work right)
                allocatedmech = False
                for stuff in reactionID:
                    if reactID == stuff[0]:
                        rmech = stuff[1]
                        allocatedmech = True

                # shouldn't exicute but just in caseTseriesKmeansloader(
                if not allocatedmech:
                    print("Reactmech not allocated for " + str(reactID))
                    rmech = None

                mechaccept.append([reactID,rmech])

    except (Exception, psycopg2.Error) as error:
        print("error,", error)
    finally:
        # This is needed for the
        if (connection):
            cursor.close()
            connection.close()

    # insert the values into sql

    return tots,mechaccept

def sqlharmcol(name):

    if name == -1: # use the total current
        yaya = """SELECT "TotalCurrent","Reaction_ID" FROM "HarmTab" WHERE "Reaction_ID" IN %s"""
    elif name == 0:
        yaya = """SELECT "HarmCol0","Reaction_ID" FROM "HarmTab" WHERE "Reaction_ID" IN %s"""
    elif name == 1:
        yaya = """SELECT "HarmCol1","Reaction_ID" FROM "HarmTab" WHERE "Reaction_ID" IN %s"""
    elif name == 2:
        yaya = """SELECT "HarmCol2","Reaction_ID" FROM "HarmTab" WHERE "Reaction_ID" IN %s"""
    elif name == 3:
        yaya = """SELECT "HarmCol3","Reaction_ID" FROM "HarmTab" WHERE "Reaction_ID" IN %s"""
    elif name == 4:
        yaya = """SELECT "HarmCol4","Reaction_ID" FROM "HarmTab" WHERE "Reaction_ID" IN %s"""
    elif name == 5:
        yaya = """SELECT "HarmCol5","Reaction_ID" FROM "HarmTab" WHERE "Reaction_ID" IN %s"""
    elif name == 6:
        yaya = """SELECT "HarmCol6","Reaction_ID" FROM "HarmTab" WHERE "Reaction_ID" IN %s"""
    elif name == 7:
        yaya = """SELECT "HarmCol7","Reaction_ID" FROM "HarmTab" WHERE "Reaction_ID" IN %s"""
    elif name == 8:
        yaya = """SELECT "HarmCol8","Reaction_ID" FROM "HarmTab" WHERE "Reaction_ID" IN %s"""
    else:
        print("Error: to many harmonics have been requested in function sqlharmcol")
        exit()

    return yaya

def KnearestN(data,n_clusters,cpu_workers):
    scaler = TimeSeriesScalerMinMax(value_range=(0., 1.))  # Rescale time series
    data = scaler.fit_transform(data)

    knn = KNeighborsTimeSeries(n_neighbors=int(len(data)/3), metric="dtw", n_jobs=cpu_workers,verbose=True )
    knn.fit(data)
    dists, ind = knn.kneighbors(data)

    print("dists")
    print(dists)
    print("indices")
    print(ind)

    print("graph")
    A = knn.kneighbors_graph(data)
    print(A.toarray())

    return dists,ind, A
def TseriesKmeansloader(data,current,fileloc):
    # modified for system
    if current == -1 or current == 0:
        scaler = TimeSeriesScalerMinMax(value_range=(-1., 1.))
        data = scaler.fit_transform(data)
        # below has been placed here as it doesn't seem to be helping the optimisation of the grouping
        currentdata = TimeSeriesScalerMeanVariance().fit_transform(data)
    else:
        scaler = TimeSeriesScalerMinMax(value_range=(0., 1.))  # Rescale time series
        currentdata = scaler.fit_transform(data)

    ks = TimeSeriesKMeans().from_hdf5(fileloc+'/trainedtimeseries'+str(current) +'.hdf5')

    y_pred = ks.predict(currentdata)

    return y_pred

def TseriesKmeans(data,n_clusters,cpu_workers,model,reactionmech,current,n_init,filename,metric):

   # modified for system
    if current == -1 or current == 0:
        scaler = TimeSeriesScalerMinMax(value_range=(-1., 1.))
        data = scaler.fit_transform(data)
        # below has been placed here as it doesn't seem to be helping the optimisation of the grouping
        currentdata = TimeSeriesScalerMeanVariance().fit_transform(data)
    else:
        scaler = TimeSeriesScalerMinMax(value_range=(0., 1.))  # Rescale time series
        currentdata = scaler.fit_transform(data)

    sz = len(currentdata)
    #this is here to save memory
    del data

    # kShape clustering
    if metric[1] != 0: # if this isn't zero system uses sakoe_chiba
        ks = TimeSeriesKMeans(n_clusters=n_clusters, max_iter=6,metric=metric[0],
                      n_init=n_init,  tol=0.001, #dtw_inertia=True,
                          n_jobs=cpu_workers)
    else:
        ks = TimeSeriesKMeans(n_clusters=n_clusters, max_iter=6, metric=metric[0],
                              metric_params={"global_constraint": "sakoe_chiba", "sakoe_chiba_radius": metric[1]},
                              n_init=n_init, tol=0.001,  # dtw_inertia=True,
                              n_jobs=cpu_workers)

    #ks.fit(currentdata)
   # change made here might of been doubling up
    y_pred = ks.fit_predict(currentdata)


    ks.to_hdf5(filename+'/trainedtimeseries'+str(current) +'.hdf5')

    nummechstore = []
    i = 0
    for values in reactionmech:
        nummechstore.append(i)
        i += 1

    datay = []
    for id in model:
        i = 0
        for values in reactionmech:
            if values == id:
                datay.append(nummechstore[i])
            i += 1

    dataplot = []
    dataploty = []
    for yi in range(n_clusters):
        i = 0
        j = 0
        while j != 10 and i != len(y_pred):
            if yi == y_pred[i]:
                dataplot.append(currentdata[i])
                dataploty.append(yi)
                j += 1
            i += 1

    plt.figure(figsize=(24, 20))
    for yi in range(n_clusters):
        plt.subplot(n_clusters, 1, 1 + yi)
        plt.plot(ks.cluster_centers_[yi].ravel(), "r-")
        i = 0
        for xx in dataplot:
            if dataploty[i] == yi:
                plt.plot(xx.ravel(), "k-", alpha=.2)

            i += 1

        plt.title("Cluster %d" % (yi + 1))

    for yi in range(n_clusters):
        plt.subplot(n_clusters, 1, 1 + yi)
        i = 0
        for xx in dataplot:
            if dataploty[i] == yi:
                #plt.plot(xx.ravel(), "k-")
                break
            i += 1

        plt.title("Cluster %d" % (yi + 1))

    plt.tight_layout()
    #plt.show()
    plt.savefig(filename+'/'+"testTimeSeriesKmean"+str(current)+ str(n_clusters) + ".png")
    plt.close()

    plt.figure()
    for yi in range(n_clusters):
        plt.plot(ks.cluster_centers_[yi].ravel())
    plt.savefig(filename+'/'+"clustermeans.png")
    plt.close()

    return y_pred

def svm(data,model,reactmech,cpu_workers):

    nummechstore = []
    i = 0
    for values in reactmech:
        nummechstore.append(i)
        i += 1

    datay = []
    for id in model:
        i = 0
        for values in reactmech:
            if values == id:
                datay.append(nummechstore[i])
            i += 1

    data = TimeSeriesScalerMinMax().fit_transform(data)

    clf = TimeSeriesSVC(kernel="gak", gamma=.1,verbose=True,n_jobs=cpu_workers)
    clf.fit(data, datay)
    print("Correct classification rate:", clf.score(data, datay))

    n_classes = len(reactmech)

    plt.figure()
    support_vectors = clf.support_vectors_
    for i, cl in enumerate(set(datay)):
        plt.subplot(n_classes, 1, i + 1)
        plt.title("Support vectors for class %d" % cl)
        for ts in support_vectors[i]:
            plt.plot(ts.ravel())

    plt.tight_layout()
    plt.show()
    plt.savefig("testsvm.png")
    return

def harmlabeltochar(mechaccept,y_predict,harmnum):
    # labels start at 0
    label = []
    i = 0
    for mechs in mechaccept:
        num = y_predict[i]

        # gets the first letter in two part label
        s1 = int(num/26)
        s1 = chr(ord('a')+s1)

        # GETS THE SECOND NUMBER
        s2 = num%26
        s2 = chr(ord('a')+s2)

        char = "H" + str(int(harmnum)) + s1 + s2

        label.append([mechs[0],char,mechs[1]]) #ReactID, label
        i += 1

    return label

def sqlharmlabel(name):

    if name == -1: # use the total current
        yaya = """UPDATE "ReactionClass" SET "TotalCurrent" = %(label)s WHERE "Reaction_ID" = %(Reaction_ID)s"""
    elif name == 0:
        yaya = """UPDATE "ReactionClass" SET "HarmCol0" = %(label)s WHERE "Reaction_ID" = %(Reaction_ID)s"""
    elif name == 1:
        yaya = """UPDATE "ReactionClass" SET "HarmCol1" = %(label)s WHERE "Reaction_ID" = %(Reaction_ID)s"""
    elif name == 2:
        yaya = """UPDATE "ReactionClass" SET "HarmCol2" = %(label)s WHERE "Reaction_ID" = %(Reaction_ID)s"""
    elif name == 3:
        yaya = """UPDATE "ReactionClass" SET "HarmCol3" = %(label)s WHERE "Reaction_ID" = %(Reaction_ID)s"""
    elif name == 4:
        yaya = """UPDATE "ReactionClass" SET "HarmCol4" = %(label)s WHERE "Reaction_ID" = %(Reaction_ID)s"""
    elif name == 5:
        yaya = """UPDATE "ReactionClass" SET "HarmCol5" = %(label)s WHERE "Reaction_ID" = %(Reaction_ID)s"""
    elif name == 6:
        yaya = """UPDATE "ReactionClass" SET "HarmCol6" = %(label)s WHERE "Reaction_ID" = %(Reaction_ID)s"""
    elif name == 7:
        yaya = """UPDATE "ReactionClass" SET "HarmCol7" = %(label)s WHERE "Reaction_ID" = %(Reaction_ID)s"""
    elif name == 8:
        yaya = """UPDATE "ReactionClass" SET "HarmCol8" = %(label)s WHERE "Reaction_ID" = %(Reaction_ID)s"""
    else:
        print("Error: to many harmonics have been requested in function sqlharmcol")
        exit()

    return yaya


def harmlabel(serverdata,label,current):

    string = sqlharmlabel(current)
    dic = {"Reaction_ID":-12,"label":"test"}

    try:
        connection = psycopg2.connect(user=serverdata[0],
                                      password=serverdata[1],
                                      host=serverdata[2],
                                      port=serverdata[3],
                                      database=serverdata[4])

        cursor = connection.cursor()

        reactionlabeldic = []
        for mech in label:
            dic = {"Reaction_ID": mech[0], "label": mech[1]}
            reactionlabeldic.append(dic)
        reactionlabeldic = tuple(reactionlabeldic)

        #extras.execute_values(cursor, string, reactionlabeldic)
        extras.execute_batch(cursor,string, reactionlabeldic,page_size=5000)
        #cursor.executemany(string, reactionlabeldic)

        connection.commit()  # the commit is needed to save changes

    except (Exception, psycopg2.Error) as error:
        print("error,", error)
    finally:
        # This is needed for the
        if (connection):
            cursor.close()
            connection.close()



    return


def outputlabelwritter(input_name,n_cluster,y_pred,mechaccept):

    f = open(input_name+"_output"+str(n_cluster)+".txt","w+")
    i = 0
    for values in y_pred:
        f.write(str(mechaccept[i][0]) +", " + str(mechaccept[i][1]) + ", " + str(values)+ "\n")
        i += 1
    f.close()

    return

def outputwritter(outputname,comtime,harmnum,y_pred,reactionmech,n_cluster,mechaccept):
    f = open(outputname, "a")

    f.write("Harmonic " + str(harmnum) +" Information\n\n")
    f.write("Iteration Completion Time: " + str(comtime))
    # add a date time thing here
    ydump = []
    xbayestot = []
    nlen = len(y_pred)

    for mech in reactionmech:
        yinnerdump = []
        xbayes = []
        i = 0
        for values in mechaccept:
            if mech == values[1]:
                yinnerdump.append(y_pred[i])
            i += 1
        f.write("\nlabel frequency for " + mech + "\n")
        nlen = len(yinnerdump)
        for i in range(n_cluster):
            ncount = sum(x1 == i for x1 in yinnerdump)
            xbayes.append(ncount)
            s = "%.3f" % (ncount / nlen)
            f.write("L" + str(i + 1) + " : " + s + " ")
        ydump.append(yinnerdump)
        xbayestot.append(xbayes)

    f.write("\n\n\nProbability of each label\n")
    for i in range(n_cluster):
        f.write("L" + str(i + 1) + " Frequency\n")
        x = 0
        for ii in range(len(reactionmech)):
            x += xbayestot[ii][i]

        for ii in range(len(reactionmech)):
            s = "%.3f" % (xbayestot[ii][i] / x)
            f.write(reactionmech[ii] + ": " + s + " ")
        f.write("\n")

    f.write("\n\nNumer of simulations per cluster\n")
    x = np.array(y_pred)
    for yi in range(n_cluster):
        num = (x == yi).sum()
        f.write("L" + str(yi + 1) + ": " + str(num) + "\n")
    f.write("Overall number of simulations: " + str(len(y_pred)))

    f.write("\n\n\n")

    f.close()

    return

def outputwritterfin(outputname,comtime,labels):
    f = open(outputname, "a")

    f.write("Overall Information\n\n")
    f.write("Overall Completion Time: " + str(comtime))

    f.write("\n\nNumer of simulations per Oveerall label\n")
    sum = 0

    for yi in labels:
        f.write(yi[0] + ": " + str(yi[1]) + "\n")
        sum += yi[1]
    f.write("Overall number of simulations: " + str(sum))

    return

def AClabelcollector(serverdata,mechaccept,n_cluster,harmnumber):
    try:
        connection = psycopg2.connect(user=serverdata[0],
                                      password=serverdata[1],
                                      host=serverdata[2],
                                      port=serverdata[3],
                                      database=serverdata[4])

        cursor = connection.cursor()

        qinput ="""SELECT "EXPsetrow", "ReactionMech" FROM "ExperimentalSettings" GROUP BY "EXPsetrow", "ReactionMech"  """
        cursor.execute(qinput,)
        expsettings = cursor.fetchall()

        print("TRY IN instead of = in the below eries")
        # use this querie system
        #cursor.execute(getmech,(reacttuple,))
        # sets up for the total label
        getdata = """SELECT "HarmCol0", "HarmCol1", "HarmCol2","HarmCol3","HarmCol4", "HarmCol5", "HarmCol6","HarmCol7","HarmCol8", "Reaction_ID"
                    FROM "ReactionClass" Where "Reaction_ID" IN %s"""

        # can be obtained from above but easier to get from below
        getmech = """SELECT "ReactionMech", "Reaction_ID" FROM "Simulatedparameters" Where "Reaction_ID" IN %s"""

        #cursor.execute(getdata,())
        #expsettings = cursor.fetchall()
        reacttuple = []
        for mech in mechaccept:
            #reacttuple.append((mech[0],)) I belevie this is what is required in the = situation
            reacttuple.append(mech[0])
        reacttuple = tuple(reacttuple)
        print("cunt")
        # gets the reaction mechanism
        #extras.execute_batch(cursor, getmech, reacttuple, page_size=5000)
        cursor.execute(getmech, (reacttuple,))
        #cursor.execute(getmech,(mech[0],))
        reactmechtot = cursor.fetchall()
        print("fuck")
        #gets the harm lables
        #cursor.execute(getdata,(mech[0],))
        #extras.execute_batch(cursor, getdata, reacttuple, page_size=5000)
        cursor.execute(getdata, (reacttuple,))
        harmlabelstot = cursor.fetchall()
        print("SHIT")
        #harmlabels = harmlabels[0]
        dicharmlabel = {}
        for harmlabels in harmlabelstot:
            s0 = harmlabels[harmnumber[0]]
            if s0 == None:
                s0 = "H" + str(harmnumber[0]) + "00"

            for i in harmnumber[1:]:
                s = harmlabels[i]

                if type(s) == type(None):
                    s = "H" + str(i) +"00"
                s0 += "_"+ s
            dicharmlabel.update({harmlabels[-1]:s0})
        print("blah")
        """UPDATE THE OVERALL LABEL"""
        qinput = """UPDATE "ReactionClass" SET "EXPsetrow" = %(EXPsetrow)s, "OverallLabel" = %(OverallLabel)s,
                    "traininglabels" = %(traininglabels)s, "classsetrow" = %(classsetrow)s WHERE "Reaction_ID" = %(Reaction_ID)s"""

        updatetuple = []
        for reactmech in reactmechtot:
             #classetrow holds the label for multiple training will change in future

            for inputs in expsettings:
                if inputs[1] == reactmech[0]:
                    EXPsetrow = inputs[0]
                    dic = { "traininglabels": n_cluster, "classsetrow": 1,"EXPsetrow": EXPsetrow,
                           "OverallLabel": dicharmlabel.get(reactmech[1]), "Reaction_ID": reactmech[1]}
            updatetuple.append(dic)
        updatetuple = tuple(updatetuple)
        print("blah2")

        extras.execute_batch(cursor, qinput, updatetuple, page_size=5000)
        #cursor.execute(qinput,dic) # (mech[0], EXPsetrow, mech[2],s0,n_cluster,1,)
        connection.commit()  # the commit is needed to save changes
        print("poopoo")
        # one off collects all the stuff for printing
        qinput = """SELECT "OverallLabel", COUNT(*) AS num FROM "ReactionClass" GROUP BY "OverallLabel" """
        cursor.execute(qinput)
        labelocurrannce = cursor.fetchall()
        print("poopoo2")

    except (Exception, psycopg2.Error) as error:
        print("error,", error)
    finally:
        # This is needed for the
        if (connection):
            cursor.close()
            connection.close()

    return labelocurrannce

"""DELETE below one"""
def OLDAClabelcollector(serverdata,mechaccept,n_cluster,harmnumber):
    try:
        connection = psycopg2.connect(user=serverdata[0],
                                      password=serverdata[1],
                                      host=serverdata[2],
                                      port=serverdata[3],
                                      database=serverdata[4])

        cursor = connection.cursor()

        qinput ="""SELECT "EXPsetrow", "ReactionMech" FROM "ExperimentalSettings" GROUP BY "EXPsetrow", "ReactionMech"  """
        cursor.execute(qinput,)
        expsettings = cursor.fetchall()


        # sets up for the total label
        getdata = """SELECT "HarmCol0", "HarmCol1", "HarmCol2","HarmCol3","HarmCol4", "HarmCol5", "HarmCol6","HarmCol7","HarmCol8"
                    FROM "ReactionClass" Where "Reaction_ID" = %s"""

        # can be obtained from above but easier to get from below
        getmech = """SELECT "ReactionMech" FROM "Simulatedparameters" Where "Reaction_ID" = %s"""

        dic = {"EXPsetrow":1}
        dic.update({"traininglabels": n_cluster})
        dic.update({"classsetrow": 1}) # holds the label for multiple training will change in future
        #cursor.execute(getdata,())
        #expsettings = cursor.fetchall()
        for mech in mechaccept:
            # gets the reaction mechanism
            cursor.execute(getmech,(mech[0],))
            reactmech = cursor.fetchall()[0][0]

            #gets the harm lables
            cursor.execute(getdata,(mech[0],))
            harmlabels = cursor.fetchall()

            harmlabels = harmlabels[0]

            s0 = harmlabels[harmnumber[0]]
            if s0 == None:
                s0 = "H" + str(harmnumber[0]) + "00"

            for i in harmnumber[1:]:
                s = harmlabels[i]

                if type(s) == type(None):
                    s = "H" + str(i) +"00"
                s0 += "_"+ s

            qinput = """UPDATE "ReactionClass" SET "EXPsetrow" = %(EXPsetrow)s, "OverallLabel" = %(OverallLabel)s,
                    "traininglabels" = %(traininglabels)s, "classsetrow" = %(classsetrow)s WHERE "Reaction_ID" = %(Reaction_ID)s"""
            for inputs in expsettings:
                if inputs[1] == reactmech:
                    EXPsetrow = inputs[0]
                    dic.update({"EXPsetrow": EXPsetrow})
            dic.update({"OverallLabel": s0})

            dic.update({"Reaction_ID": mech[0]})
            cursor.execute(qinput,dic) # (mech[0], EXPsetrow, mech[2],s0,n_cluster,1,)

        connection.commit()  # the commit is needed to save changes

        qinput = """SELECT "OverallLabel", COUNT(*) AS num FROM "ReactionClass" GROUP BY "OverallLabel" """
        cursor.execute(qinput)
        labelocurrannce = cursor.fetchall()

    except (Exception, psycopg2.Error) as error:
        print("error,", error)
    finally:
        # This is needed for the
        if (connection):
            cursor.close()
            connection.close()

    return labelocurrannce

def DClabelcollector(serverdata,mechaccept,n_cluster):
    try:
        connection = psycopg2.connect(user=serverdata[0],
                                      password=serverdata[1],
                                      host=serverdata[2],
                                      port=serverdata[3],
                                      database=serverdata[4])

        cursor = connection.cursor()

        qinput ="""SELECT "EXPsetrow", "ReactionMech" FROM "ExperimentalSettings" GROUP BY "EXPsetrow", "ReactionMech"  """
        cursor.execute(qinput,)
        expsettings = cursor.fetchall()

        for mech in mechaccept:

            qinput = """INSERT INTO "ReactionClass"("Reaction_ID","EXPsetrow","ReactionMech","TotalCurrent","traininglabels","classsetrow") VALUES(%s,%s,%s,%s,%s,%s)"""
            for inputs in expsettings:
                if inputs[1] == mech[2]:
                    EXPsetrow = inputs[0]
            cursor.execute(qinput, (mech[0], EXPsetrow, mech[2],mech[1],n_cluster,1,))

        connection.commit()  # the commit is needed to save changes

        qinput = """SELECT "TotalCurrent", COUNT(*) AS num FROM "ReactionClass" GROUP BY "TotalCurrent" """
        cursor.execute(qinput)
        labelocurrannce = cursor.fetchall()

    except (Exception, psycopg2.Error) as error:
        print("error,", error)
    finally:
        # This is needed for the
        if (connection):
            cursor.close()
            connection.close()

    return labelocurrannce

def ACLabelpreallocator(serverdata, reactID):
    try:
        connection = psycopg2.connect(user=serverdata[0],
                                      password=serverdata[1],
                                      host=serverdata[2],
                                      port=serverdata[3],
                                      database=serverdata[4])

        #Preallocates the rows with and ID and mechanism

        reactlistID = []
        for ids in reactID:
            reactlistID.append(tuple(ids))

        cursor = connection.cursor()
        t1 = time.time()

        qinput ="""INSERT INTO "ReactionClass"("Reaction_ID", "ReactionMech") VALUES %s  """
        extras.execute_values(cursor,qinput,reactlistID)
        connection.commit()  # the commit is needed to save changes
        print("Time for preallocation: "+ str((time.time() - t1)/60))

    except (Exception, psycopg2.Error) as error:
        print("error,", error)
    finally:
        # This is needed for the
        if (connection):
            cursor.close()
            connection.close()

    return

def labelprinterDC(filename,strlabel):

    f = open(filename+"/ReactionIDlabels.txt","w+")
    for mechs in strlabel:
        f.write(str(mechs[0])+"\t"+str(mechs[1])+"\n")
    f.close()

    return
