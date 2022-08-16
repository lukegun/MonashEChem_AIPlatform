
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
        x.append(values.strip(" "))
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

# placeholder function for loading the
def modeldataloader(modelparamterfile):

    return

def inputstrip(x):
    y = x.strip("\n\t ")
    y = y.split("#")[0]
    y = y.strip("\n\t ")
    y = y.split("=")[-1]
    y = y.strip(" \t")
    return y

def EXPsetrow_collector(serverdata,models):
    connection = psycopg2.connect(user="postgres",
                                  password="Quantum2",
                                  host="localhost",
                                  port="5432",
                                  database=serverdata[0])

    cursor = connection.cursor()

    exp_class = []

    for mech in models:

        # put in the parameers where they sould go
        try:
            # gets the EXPsetrow for reaction mechanism and experimental reaction mech
            print(mech)
            cursor.execute("""SELECT "EXPsetrow" FROM "ExperimentalSettings" WHERE "ReactionMech" =  %s AND "ACFreq1" = %s::real[] """, (mech,))
            # extracts the
            cuurr = cursor.fetchall()[0]

            # check to see if there are no dublicates
            if len(list(cuurr)) != 1:
                print('ERROR: duplicate experimental settings found in expsettings = ' + str(list(cuurr)))
                exit()
            else:
                exp_class.append([cuurr[0],mech])

        except (Exception, psycopg2.Error) as error:
            print("error message from sql_updater1,", error)
            exit()

    if (connection):
        cursor.close()
        connection.close()

    return exp_class

# split and shuffle react_class for each independant reactionMech
def suffle_splittrainratio(trainratio, react_class):
    holder = []
    for values in react_class:
        holder += values

    """There may be an issue here with shuffling a large length array"""
    Ntrain = int(round(trainratio * len(holder)))

    Ntest = int(len(holder) - Ntrain)
    np.random.shuffle(holder)  # shuffle change the order of the list
    testdata = holder[0:Ntest]
    traindata = holder[Ntest::]

    return testdata, traindata, Ntest, Ntrain

