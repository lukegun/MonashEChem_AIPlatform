import time

import psycopg2
import numpy as np

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

    supivisorclass = str(inputstrip(lines[1]))

    x = inputstrip(lines[2])
    ACCase = [x == "True" or x == "true"]
    ACCase = ACCase[0]

    x = inputstrip(lines[3])
    variate = [x == "True" or x == "true"]
    variate = variate[0]

    DNNmodel = str(inputstrip(lines[4]))  # rocket or inceptiontime

    cpu_workers = int(inputstrip(lines[5]))

    gpu_workers = int(inputstrip(lines[6]))

    deci = 2**int(inputstrip(lines[7]))

    harmdata = inputstrip(lines[8])
    harmdata = harmdata.split(",")
    x = []
    # order is [user, password, host, port, database]
    for values in harmdata:
        x.append(int(values.strip(" ")))
    harmdata = x

    modelparamterfile = str(inputstrip(lines[9]))  # rocket or inceptiontime

    trainratio = float(inputstrip(lines[10]))  # rocket or inceptiontime

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


    return serverdata, supivisorclass, ACCase, variate, DNNmodel, cpu_workers, gpu_workers, deci, harmdata, modelparamterfile, trainratio, reactionmech

# something to collect DNN model specific parameters from the
def modeldataloader(modelparamterfile):
    modeldic = {}

    return modeldic

def inputstrip(x):
    y = x.strip("\n\t ")
    y = y.split("#")[0]
    y = y.strip("\n\t ")
    y = y.split("=")[-1]
    y = y.strip(" \t")
    return y

# connect to database and extract the
def EXPsetrow_dcdata(serverdata,exp_class):

    connection = psycopg2.connect(user=serverdata[0],
                                  password=serverdata[1],
                                  host=serverdata[2],
                                  port=serverdata[3],
                                  database=serverdata[4])

    cursor = connection.cursor()

    exp_data = []

    for mech in exp_class:

        row = mech[0]

        dic = {}
        dic.update({"Expsetrow":row})
        dic.update({"mech": mech[1]})

        # put in the parameers where they sould go
        try:
            # gets the EXPsetrow for reaction mechanism and experimental reaction mech

            cursor.execute("""SELECT "Estart" FROM "ExperimentalSettings" WHERE "EXPsetrow" =  %s """, (row,))
            # extracts the
            Estart = cursor.fetchall()[0][0]

            dic.update({"Estart": Estart})

            cursor.execute(
                """SELECT "Erev" FROM "ExperimentalSettings" WHERE "EXPsetrow" =  %s """,
                (row,))
            # extracts the
            Erev = cursor.fetchall()[0][0]

            dic.update({"Erev": Erev})

            cursor.execute(
                """SELECT "cyclenum" FROM "ExperimentalSettings" WHERE "EXPsetrow" =  %s """,
                (row,))
            # extracts the
            cyclenum = cursor.fetchall()[0][0]

            dic.update({"cyclenum": cyclenum})

        except (Exception, psycopg2.Error) as error:
            print("error message from sql_updater1,", error)
            exit()

        exp_data.append(dic)

    if (connection):
        cursor.close()
        connection.close()

    return exp_data

# connect to server and get an list of all reaction IDs and paired with class of form react_class = [[ReactionID,Reactionmech]]
def ReactID_collector(serverdata,exp_class):
    connection = psycopg2.connect(user=serverdata[0],
                                  password=serverdata[1],
                                  host=serverdata[2],
                                  port=serverdata[3],
                                  database=serverdata[4])

    cursor = connection.cursor()

    react_class_tot = []

    for expset in exp_class:
        react_class = []
        # put in the parameers where they sould go
        try:
            # gets the EXPsetrow for reaction mechanism and experimental reaction mech
            cursor.execute("""SELECT "Reaction_ID","NumHarm" FROM "Simulatedparameters" WHERE "ReactionMech" =  %s AND "EXPsetrow" = %s """, (expset[1],expset[0]))
            # extracts the
            cuurr = cursor.fetchall()

            """IN FUTURE YOU CAN PROBABLY INCLUDE SOMETHING THAT INCORPURATES THE BV or MH REACTION MODEL"""

            for IDs in cuurr:
                react_class.append([IDs[0],expset[1],IDs[1]])  # of the form ReactionIDS, classification,
            react_class_tot.append(react_class)

        except (Exception, psycopg2.Error) as error:
            print("error message from sql_updater2,", error)
            exit()

    if (connection):
        cursor.close()
        connection.close()


    return react_class_tot

def EXPsetrow_collector(serverdata,models):
    connection = psycopg2.connect(user=serverdata[0],
                                  password=serverdata[1],
                                  host=serverdata[2],
                                  port=serverdata[3],
                                  database=serverdata[4])

    cursor = connection.cursor()

    exp_class = []

    for mech in models:

        # put in the parameers where they sould go
        try:
            # gets the EXPsetrow for reaction mechanism and experimental reaction mech
            cursor.execute("""SELECT "EXPsetrow" FROM "ExperimentalSettings" WHERE "ReactionMech" =  %s""", (mech,))
            # extracts the
            cuurr = cursor.fetchall()[0]

            # check to see if there are no dublicates
            if len(list(cuurr)) != 1:
                print('ERROR: duplicate experimental settings found in expsettings = ' + str(list(cuurr)))
                exit()
            else:
                exp_class.append([cuurr[0],mech])

        except (Exception, psycopg2.Error) as error:
            print("error message from sql_updater3,", error)
            exit()

    if (connection):
        cursor.close()
        connection.close()

    return exp_class

# count number for each reaction model and check if same
def Narray_count(react_class):
    Narray = []
    x = 0
    for arrays in react_class:
        Narray.append(len(arrays))
        x += len(arrays)

    if all(Narray[0] == rest for rest in Narray):
        Narray = Narray[0]
    else:
        print("Error: length of tests do not equal accross all classifications")
    Narray = x
    return Narray

def sqlcurrentcollector(serverdata,current, reactionID,deci):

    try:
        connection = psycopg2.connect(user=serverdata[0],
                                      password=serverdata[1],
                                      host=serverdata[2],
                                      port=serverdata[3],
                                      database=serverdata[4])

        cursor = connection.cursor()

        #yaya = sqlharmcol(current)
        yaya = """SELECT "Reaction_ID", "TotalCurrent" FROM "HarmTab" WHERE "Reaction_ID" IN %s"""

        tots = []
        mechaccept = []

        # breaks all of this up so that there isn't like 2 times the database present at one time
        list = []
        i = 1
        for ID in reactionID:
            list.append(ID[0])
            i += 1

        listtot = tuple(list)

        cursor.execute(yaya, (listtot,))

        xtotdata = cursor.fetchall()

        for ID in reactionID:

            for i in range(len(list)):
                if ID[0] == xtotdata[i][0]:
                    xtot = xtotdata[i][1]
                    break

            if type(xtot) is type(None):

                pass
            else:
                xtot = np.array(xtot)
                nlen = len(xtot)
                Ndec = int(nlen / deci)
                xtot = xtot[::Ndec]

                xtot = xtot / max(xtot)
                tots.append(xtot)
                mechaccept.append([ID[0],ID[1]])


    except (Exception, psycopg2.Error) as error:
        print("error,", error)
    finally:
        # This is needed for the
        if (connection):
            cursor.close()
            connection.close()

    # insert the values into sql

    return tots,mechaccept

def ACsqlcurrentcollector(serverdata, harmdata, reactionID, deci,DNNmodel):
    try:
        connection = psycopg2.connect(user=serverdata[0],
                                      password=serverdata[1],
                                      host=serverdata[2],
                                      port=serverdata[3],
                                      database=serverdata[4])

        cursor = connection.cursor()

        if DNNmodel == "rocket":
            Rocket = True
        else:
            Rocket = False

        # breaks all of this up so that there isn't like 2 times the database present at one time
        listtot1 = []
        listtot2 = []
        list = []
        list2 = []
        i = 1
        for ID in reactionID:
            if i%3000 == 0:
                listtot1.append(tuple(list))
                list= []
                listtot2.append(list2)
                list2 = []
            list.append(ID[0])
            list2.append(ID)
            i += 1

        # exception so we don't put in empty arrays
        if i%3000 != 0:
            listtot2.append(list2)
            listtot1.append(tuple(list))

        tots = []
        mechaccept = []
        j = 0
        for listtot in listtot1:

            t1 = time.time()
            yaya = """SELECT "Reaction_ID","HarmCol0","HarmCol1","HarmCol2","HarmCol3","HarmCol4","HarmCol5","HarmCol6","HarmCol7","HarmCol8"  FROM "HarmTab" WHERE "Reaction_ID" IN %s"""
            cursor.execute(yaya, (listtot,))
            xtotbigtot = cursor.fetchall()
            print(time.time() - t1)

            for ID in listtot2[j]: # itterate and search for arrays allocated in above section for use
                for i in range(len(listtot2[j])):
                    if ID[0] == xtotbigtot[i][0]:
                        xtotbig = xtotbigtot[i][1:]
                        break

                if Rocket:
                    dic = {}
                else:
                    dic = np.zeros((deci,len(harmdata)))

                mechaccept.append([ID[0], ID[1]])

                i = 0
                breakout = False
                for current in harmdata:

                    xtot = xtotbig[current]

                    if type(xtot) is type(None):
                        # if nontype is returned it means that max value has been reached for this React_ID and we can move onto next one
                        breakout = True
                        break
                    else:
                        xtot = np.array(xtot)
                        nlen = len(xtot)
                        Ndec = int(nlen / deci)
                        xtot = xtot[::Ndec]
                        xtot = xtot / max(xtot)

                        if Rocket:
                            dic.update({str(current): xtot})
                        else:
                            dic[:,i] = xtot[:]
                    i += 1

                # replaces Nan values with empty arrays
                if breakout and Rocket:
                    for l in range(current,harmdata[-1] + 1):
                        dic.update({str(l): np.zeros(deci)})

                tots.append(dic)
            j += 1

    except (Exception, psycopg2.Error) as error:
        print("error in ACsqlcurrentcollector,", error)
    finally:
        # This is needed for the
        if (connection):
            cursor.close()
            connection.close()

    # insert the values into sql

    return tots, mechaccept


def sqlharmcol(name):

    if name == -1: # use the total current
        yaya = """SELECT "TotalCurrent" FROM "HarmTab" WHERE "Reaction_ID" = %s"""
    elif name == 0:
        yaya = """SELECT "HarmCol0" FROM "HarmTab" WHERE "Reaction_ID" = %s"""
    elif name == 1:
        yaya = """SELECT "HarmCol1" FROM "HarmTab" WHERE "Reaction_ID" = %s"""
    elif name == 2:
        yaya = """SELECT "HarmCol2" FROM "HarmTab" WHERE "Reaction_ID" = %s"""
    elif name == 3:
        yaya = """SELECT "HarmCol3" FROM "HarmTab" WHERE "Reaction_ID" = %s"""
    elif name == 4:
        yaya = """SELECT "HarmCol4" FROM "HarmTab" WHERE "Reaction_ID" = %s"""
    elif name == 5:
        yaya = """SELECT "HarmCol5" FROM "HarmTab" WHERE "Reaction_ID" = %s"""
    elif name == 6:
        yaya = """SELECT "HarmCol6" FROM "HarmTab" WHERE "Reaction_ID" = %s"""
    elif name == 7:
        yaya = """SELECT "HarmCol7" FROM "HarmTab" WHERE "Reaction_ID" = %s"""
    elif name == 8:
        yaya = """SELECT "HarmCol8" FROM "HarmTab" WHERE "Reaction_ID" = %s"""
    else:
        print("Error: to many harmonics have been requested in function sqlharmcol")
        exit()

    return yaya

# collects the clustering algorithm data from the system
def ACreactclusterupdate(react_class,serverdata,num = 25):

    try:
        connection = psycopg2.connect(user=serverdata[0],
                                      password=serverdata[1],
                                      host=serverdata[2],
                                      port=serverdata[3],
                                      database=serverdata[4])

        cursor = connection.cursor()

        yaya = """SELECT "OverallLabel", COUNT(*) AS num FROM "ReactionClass"  GROUP BY "OverallLabel"  HAVING COUNT(*) >%s """

        cursor.execute(yaya,(num,))

        Groupclass = cursor.fetchall()

        newreact_class = []
        reactmech = []
        for reactionmechlabels in react_class:
            newreact = []
            for ID in reactionmechlabels:
                cursor.execute(yaya, (ID[0],))

                xclass = cursor.fetchall()[0][0]

                newreact.append([ID[0],xclass])

                if len(reactmech) == 0:
                    reactmech.append(xclass)
                else:
                    exist = True
                    for reacts in reactmech:
                        if reacts == xclass:
                            exist = False
                    if exist:
                        reactmech.append(xclass)
            newreact_class.append(newreact)

    except (Exception, psycopg2.Error) as error:
        print("error,", error)
    finally:
        # This is needed for the
        if (connection):
            cursor.close()
            connection.close()

    return  Groupclass

# collects the clustering algorithm data from the system
def DCreactclusterupdate(react_class,serverdata):

    try:
        connection = psycopg2.connect(user=serverdata[0],
                                      password=serverdata[1],
                                      host=serverdata[2],
                                      port=serverdata[3],
                                      database=serverdata[4])

        cursor = connection.cursor()

        yaya = """SELECT "TotalCurrent" FROM "ReactionClass" WHERE "Reaction_ID" = %s"""

        newreact_class = []
        reactmech = []
        for reactionmechlabels in react_class:
            newreact = []
            for ID in reactionmechlabels:
                cursor.execute(yaya, (ID[0],))

                xclass = cursor.fetchall()[0][0]

                newreact.append([ID[0],xclass])

                if len(reactmech) == 0:
                    reactmech.append(xclass)
                else:
                    exist = True
                    for reacts in reactmech:
                        if reacts == xclass:
                            exist = False
                    if exist:
                        reactmech.append(xclass)
            newreact_class.append(newreact)

    except (Exception, psycopg2.Error) as error:
        print("error,", error)
    finally:
        # This is needed for the
        if (connection):
            cursor.close()
            connection.close()

    return newreact_class, reactmech

# collects the clustering algorithm data from the system
def ACreactclusterupdate2(serverdata,groupdic):

    try:
        connection = psycopg2.connect(user=serverdata[0],
                                      password=serverdata[1],
                                      host=serverdata[2],
                                      port=serverdata[3],
                                      database=serverdata[4])

        cursor = connection.cursor()

        yaya = """SELECT "Reaction_ID" FROM "ReactionClass" WHERE "OverallLabel" = %s """

        newreact_class = []
        for reactionmechlabels,innerlabel in groupdic.items():
            innclass = []
            for labels in innerlabel:
                # extracts the harmnum from the labels
                harmnum = 8 # assume its 8 long
                harmnum0 = labels.split('_H') # gets the harmonic number of the label
                for stuff in harmnum0:
                    if stuff[1:3] == "00":
                        harmnum = int(stuff[0]) - 1
                        break

                cursor.execute(yaya,(labels,))
                xclass = cursor.fetchall()
                for values in xclass:
                    innclass.append([values[0],reactionmechlabels,harmnum])

            newreact_class.append(innclass)

    except (Exception, psycopg2.Error) as error:
        print("error,", error)
    finally:
        # This is needed for the
        if (connection):
            cursor.close()
            connection.close()

    return newreact_class

def NN_ILgenerator_dc_exp(current,Nx,Ny,minI,maxI):
    n_i = int(Ny)
    n_e = int(Nx)
    n_sig_fig = 4

    Estart = 0.67
    Erev = -0.67
    #print("WARNING Estaet: " + str(Estart) +" and " + "Erev: " + str(Erev))
    Ncycles = 1
    dpoints = len(current)

    for i in range(int(Ncycles)):
        DC1 = np.linspace(Estart,Erev,int(dpoints/(2*Ncycles)))
        DC2 = np.linspace(Erev, Estart, int(dpoints / (2 * Ncycles)))
        if i == 0:
            DC = np.append(DC1,DC2)
        else:
            x = np.append(DC1, DC2)
            DC = np.append(DC,x)

    e_dc = DC

    min_i = minI
    max_i = maxI
    del_i = (max_i - min_i) / (n_i - 1)
    min_e = e_dc.min()
    max_e = e_dc.max()
    del_e = (max_e - min_e) / (n_e - 1)
    i_bin = [int((current[i] - min_i) / del_i) for i in range(0, len(current))]
    e_bin = [int((e_dc[i] - min_e) / del_e) for i in range(0, len(e_dc))]
    # build structure
    ml_output = np.zeros((n_i, n_e))
    # increment for each bin member
    for i in range(0, len(e_dc)):
        ml_output[i_bin[i], e_bin[i]] += 1
    # reshape to single array
    ml_output = ml_output.reshape(n_i * n_e)
    # scale
    max_count = ml_output.max()
    # / max, * 10^sig fig
    ml_output = [(10 ** n_sig_fig) * ml_output[i] / max_count for i in range(0, len(ml_output))]
    # to int via ceiling
    ml_output = np.ceil(ml_output)
    # / 10^sig fig
    ml_output = [round(ml_output[i] / (10 ** n_sig_fig), n_sig_fig) for i in range(0, len(ml_output))]
    ml_output = np.array(ml_output)
    # save to text file
    ml_reshaped = ml_output.reshape((n_i, n_e))

    return ml_reshaped