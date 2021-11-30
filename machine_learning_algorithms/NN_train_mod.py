"""
Modules for the development of NN_train for the aplication of CNN to the application of FTACV

Arthor: Luke Gundry

"""

import psycopg2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras
import random
from random import shuffle # mixing up or currently ordered data that might lead our network astray in training.
import os
import datetime
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool # for collecting the data

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv1D, Activation
from tensorflow.keras.layers import AveragePooling2D, SpatialDropout2D
from tensorflow.keras.layers import Input, concatenate, Add # this is for layered inputs
from tensorflow.keras import mixed_precision

# connect to database and extract the
def EXPsetrow_collector(serverdata,models,ACcomp):
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
            print(ACcomp)
            print(mech)
            cursor.execute("""SELECT "EXPsetrow" FROM "ExperimentalSettings" WHERE "ReactionMech" =  %s AND "ACFreq1" = %s::real[] """, (mech,ACcomp))
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

# connect to database and extract the
def EXPsetrow_dcdata(serverdata,exp_class):
    connection = psycopg2.connect(user="postgres",
                                  password="Quantum2",
                                  host="localhost",
                                  port="5432",
                                  database=serverdata[0])

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

    connection = psycopg2.connect(user="postgres",
                                  password="Quantum2",
                                  host="localhost",
                                  port="5432",
                                  database=serverdata[0])

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

# count number for each reaction model and check if same
def Narray_count(react_class):
    Narray = []
    for arrays in react_class:
        Narray.append(len(arrays))

    if all(Narray[0] == rest for rest in Narray):
        Narray = Narray[0]
    else:
        print("Error: length of tests do not equal accross all classifications")
        exit()
    return Narray

# split and shuffle react_class for each independant reactionMech
def suffle_split(testratio,Narray, react_class):
    Ntest = int(round(testratio*Narray))
    Ntrain = int(Narray - Ntest)
    holder = []
    for values in react_class:
        holder += values
    """There may be an issue here with shuffling a large length array"""
    np.random.shuffle(holder)  # shuffle change the order of the list
    testdata = holder[0:Ntest]
    traindata = holder[Ntest::]

    return testdata, traindata, Ntest, Ntrain

# batch setter for NN
def NN_batchsetter(testdata,batchsize):

    totarray = []
    j = 0

    for i in range(int(len(testdata)//batchsize)):  #throws away last not complete batch
        array = []
        for k in range(batchsize):
            array.append(testdata[j])
            j += 1
        totarray.append(array)

    return totarray

# extracts the reaction IDs to format for processing
def NN_setter(datain,harmnum):
    data_ID = []
    data_mech = []
    data = []
    s = ''
    for inputs in datain:
        if inputs[2] < harmnum:
            pass
        else:
            data_ID.append(inputs[0])
            data_mech.append(inputs[1])
            data.append(
                [inputs[0], inputs[1]])  # required as i don't trust other methods to be stable on parallelelisation
    return data, data_mech, data_ID

def DC_NN_setter(datain):
    data_ID = []
    data_mech = []
    data = []
    s = ''
    for inputs in datain:

        data_ID.append(inputs[0])
        data_mech.append(inputs[1])
        data.append(
                [inputs[0], inputs[1]])  # required as i don't trust other methods to be stable on parallelelisation
    return data, data_mech, data_ID

# If loop for selecting the HarmCol as im lazy a fuck
def sql_colselect(harmnum):

    if harmnum == 0:
        tab = """SELECT "HarmCol0" FROM "HarmTab" WHERE "Reaction_ID" =  %s"""
    elif harmnum == 1:
        tab = """SELECT "HarmCol1" FROM "HarmTab" WHERE "Reaction_ID" =  %s"""
    elif harmnum == 2:
        tab = """SELECT "HarmCol2" FROM "HarmTab" WHERE "Reaction_ID" =  %s"""
    elif harmnum == 3:
        tab = """SELECT "HarmCol3" FROM "HarmTab" WHERE "Reaction_ID" =  %s"""
    elif harmnum == 4:
        tab = """SELECT "HarmCol4" FROM "HarmTab" WHERE "Reaction_ID" =  %s"""
    elif harmnum == 5:
        tab = """SELECT "HarmCol5" FROM "HarmTab" WHERE "Reaction_ID" =  %s"""
    elif harmnum == 6:
        tab = """SELECT "HarmCol6" FROM "HarmTab" WHERE "Reaction_ID" =  %s"""
    elif harmnum == 7:
        tab = """SELECT "HarmCol7" FROM "HarmTab" WHERE "Reaction_ID" =  %s"""
    elif harmnum == 8:
        tab = """SELECT "HarmCol8" FROM "HarmTab" WHERE "Reaction_ID" =  %s"""
    elif harmnum == 9:
        tab = """SELECT "HarmCol9" FROM "HarmTab" WHERE "Reaction_ID" =  %s"""
    elif harmnum == 10:
        tab = """SELECT "HarmCol10" FROM "HarmTab" WHERE "Reaction_ID" =  %s"""
    elif harmnum == 11:
        tab = """SELECT "HarmCol11" FROM "HarmTab" WHERE "Reaction_ID" =  %s"""
    elif harmnum == 12:
        tab = """SELECT "HarmCol12" FROM "HarmTab" WHERE "Reaction_ID" =  %s"""
    else:
        print("ERROR: harmonic number to large tat what is put in function sql_colselect")
        exit()
    return tab

def NN_codegenerator(inputdata, model_numcode):

    x = []
    for stuff in inputdata:
        stuff[1] = model_numcode.get(stuff[1])
        x.append(stuff)
    return x

class Machine_learning_class(keras.utils.Sequence):
    # initialisation
    def __init__(self,inputdata,harmnum,batchsize=32,dim=(100,100),serverdata=["testdb"],n_classes = 1000,n_channels =1,model_numcode = {"E":0},cpu_number = 1,exp_data = {}):
        self.harmnum = harmnum # which number harmonic we are identifying
        self.batch_size = batchsize # numer of data points we are taking
        self.imagedimensions = dim
        self.serverdata = serverdata
        inputdata =  NN_codegenerator(inputdata, model_numcode)
        self.batch_indexes = NN_batchsetter(inputdata, batchsize)    # Generate indexes of the batch
        self.indices = len(inputdata)
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.cpu_number = cpu_number
        self.exp_data = exp_data

    # THIS OVERWRITE THINGS IN TH KERAS SEQUENCE CLASS
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        t1 = time.time()
        list_IDs_temp = self.batch_indexes[index]

        # Generate data
        X, y = self.FTACV_CNN_generator(list_IDs_temp)
        print(" Batch Time: " + str(time.time() - t1))
        #dataset = self.FTACV_CNN_generator(list_IDs_temp)

        #print(np.random.randint(0,100000))
        #print(dataset)
        #print(y)

        return np.array(X), np.array(y)

    def __len__(self):
        # Denotes the number of batches per epoch
        return self.indices  // self.batch_size

    # Generator for the machine learning model
    def FTACV_CNN_generator(self, batchdata): # harmnum is the harmonic we are training for

        # Initialization
        X = np.empty((self.batch_size, self.imagedimensions[1], self.imagedimensions[0], self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        #with Pool(processes=Ncore) as p:

            #FIT = p.map(run, listin)
        react_ID = []
        for i, inputs in enumerate(batchdata):
            # choose random index in features
            react_ID.append( inputs[0])
            ##image = self.image_collector(react_ID) # inputs[1] = Classification
            #X[i,:,:,:] = image[0,:,:,:]
            y[i] = inputs[1]

        run = self.image_collector

        with Pool(processes=self.cpu_number) as p:

            Batchimage = p.map(run, react_ID)

        for i in range(self.batch_size):
            X[i, :, :, :] = Batchimage[i][0, :, :, :]

        #Convert to  a tf dataset
        #dataset = tf.data.Dataset.from_tensor_slices((X, y))

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

        # Generator for the machine learning model
    def FTACV_CNN_dataset(self, batchdata):  # harmnum is the harmonic we are training for
        # Initialization
        X = np.empty((self.batch_size, self.imagedimensions[1], self.imagedimensions[0], self.n_channels))
        y = np.empty((self.batch_size), dtype=int)
        for i, inputs in enumerate(batchdata):
            # choose random index in features
            react_ID = inputs[0]
            image = self.image_collector(react_ID)  # inputs[1] = Classification
            X[i, :, :, :] = image[0, :, :, :]
            y[i] = inputs[1]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)


    # connect to server and get an list of all reaction IDs and paired with class of form react_class = [[ReactionID,Reactionmech]]
    def image_collector(self,react_ID):

        connection = psycopg2.connect(user="postgres",
                                      password="Quantum2",
                                      host="localhost",
                                      port="5432",
                                      database=self.serverdata[0])

        cursor = connection.cursor()

        #react_class_tot = []
        #react_class = []
        # put in the parameers where they sould go

        try:
            #print(react_ID)
            # gets the EXPsetrow for reaction mechanism and experimental reaction mech
            tab = sql_colselect(self.harmnum) # this is here as im lazy and it works
            cursor.execute(tab, (react_ID,))
                # extracts the
            cuurr = np.array(cursor.fetchall()[0][0])

            #format_txt = '%.' + str(4) + 'f' this is for saving the dta to a csv
            #harmonic_data = np.frombuffer(cuurr[0][self.harmnum]) .reshape(self.imagedimensions[0], self.imagedimensions[1])
            if self.harmnum != 0:
                harmonic_data = NN_ILgenerator_harmflip(cuurr,self.imagedimensions[0], self.imagedimensions[1])
            else:
                """print("WARNING: DC works but is cheap and assumes that Estart EEnd ar all same")"""
                harmonic_data = NN_ILgenerator_dc(cuurr, self.imagedimensions[0], self.imagedimensions[1],self.exp_data[0])
            #harmonic_data = NN_ILgenerator(cuurr, self.imagedimensions[0], self.imagedimensions[1]) previous

            """
            Required to get into a format that is easy for Keras (and Tensorflow underneath) to use. Format is:

            (n_images, IMAGE_X, IMAGE_Y, n_channels)
                
            where n_images is the number of images to stack on top of each other in a single batch or the entire training 
            set, the image has resolution X,Y and the number of channels is 1 for black and white or 3 for RGB colour images.            
            """

            x_train = harmonic_data.reshape(1,self.imagedimensions[1], self.imagedimensions[0],1)

        except (Exception, psycopg2.Error) as error:
            print("error message from sql_updater3,", error)
            exit()

        if (connection):
            cursor.close()
            connection.close()

        return x_train

# training data passer
def FTACV_CNN_trainer(inputdata,batch_size,n_channels,imagedimensions,serverdata,harmnum,model_numcode): # harmnum is the harmonic we are training for

    inputdata = NN_codegenerator(inputdata, model_numcode)
    random.shuffle(inputdata)
    batchdata = inputdata[0:batch_size]
    # Initialization
    X = np.empty((batch_size, *imagedimensions , n_channels))
    y = np.empty((batch_size), dtype=int)

    i = 0
    for inputs in batchdata:
        # choose random index in features
        react_ID = inputs[0]
        image = trainimage_collector(react_ID,serverdata,harmnum,imagedimensions) # inputs[1] = Classification
        X[i,:,:,:] = image[0,:,:,:]
        y[i] = inputs[1]
        i += 1

    return (X, y)

def trainimage_collector(react_ID,serverdata,harmnum,imagedimensions):

    connection = psycopg2.connect(user="postgres",
                                      password="Quantum2",
                                      host="localhost",
                                      port="5432",
                                      database=serverdata[0])

    cursor = connection.cursor()

    # react_class_tot = []
    # react_class = []
    # put in the parameers where they sould go

    try:
    # print(react_ID)
    # gets the EXPsetrow for reaction mechanism and experimental reaction mech
        tab = sql_colselect(harmnum)  # this is here as im lazy and it works
        cursor.execute(tab, (react_ID,))
    # extracts the
        cuurr = np.array(cursor.fetchall()[0][0])

    # format_txt = '%.' + str(4) + 'f' this is for saving the dta to a csv
    # harmonic_data = np.frombuffer(cuurr[0][self.harmnum]) .reshape(self.imagedimensions[0], self.imagedimensions[1])
        harmonic_data = NN_ILgeneratorflip(cuurr, imagedimensions[0], imagedimensions[1])
        #harmonic_data = NN_ILgenerator(cuurr, imagedimensions[0], imagedimensions[1]) previous

        x_train = harmonic_data.reshape(1, imagedimensions[0], imagedimensions[1], 1)

    except (Exception, psycopg2.Error) as error:
        print("error message from sql_updater4,", error)
        exit()

    if (connection):
        cursor.close()
        connection.close()

    return x_train

# takes the expermental current and generates the images into a image for use in the NN as  greyscal image
def NN_ILgenerator(current,Nx,Ny):
    #tim3 = time.time()
    n_i = Ny
    n_e = Nx
    n_sig_fig = 4

    e_dc = np.linspace(0,100,len(current))

    min_i = current.min()
    max_i = current.max()
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
    #format_txt = '%.' + str(n_sig_fig) + 'f'
    #np.savetxt("harmtestpost.csv", ml_reshaped[::-1,:], delimiter=",", fmt=format_txt)

    #binarydata = ml_reshaped[::-1,:].tobytes()      # This weird correction is here so that time series is in the right order
    #print(time.time() - tim3)

    return ml_reshaped

def NN_ILgenerator_dc(current,Nx,Ny,data):
    n_i = int(Ny )
    n_e = int(Nx)
    n_sig_fig = 4

    Estart = float(data.get("Estart"))
    Erev = float(data.get("Erev"))
    Ncycles = float(data.get("cyclenum"))
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

    min_i = min(current)
    max_i = max(current)
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

def NN_ILgenerator_dc_exp(current,Nx,Ny):
    n_i = int(Ny )
    n_e = int(Nx)
    n_sig_fig = 4

    print("WARNING: SET VOLTAGE FOR DC IMAGE")
    Estart = 0.67
    Erev = -0.67
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

    min_i = min(current)
    max_i = max(current)
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

def NN_ILgenerator_harmflip(current,Nx,Ny):
    #tim3 = time.time()
    n_i = int(Ny * 0.5)
    n_e = int(Nx * 2)
    n_sig_fig = 4

    e_dc = np.linspace(0,100,len(current))

    min_i = min( current)
    max_i = max(current)
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

    forward = ml_reshaped[:,0:int(n_e/2)]
    reverse = ml_reshaped[:, int(n_e/2):]

    ml_reshaped = np.concatenate((np.flip(forward),reverse),axis=0)

    return ml_reshaped

def NN_ILgeneratorflip(current,Nx,Ny):
    #tim3 = time.time()
    n_i = int(Ny*0.5)
    n_e = int(Nx*2)
    n_sig_fig = 4

    e_dc = np.linspace(0,100,len(current))

    min_i = current.min()
    max_i = current.max()
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

    forward = ml_reshaped[:,0:int(n_i/2)]
    reverse = ml_reshaped[:, int(n_i/2):]

    ml_reshaped = np.concatenate((np.flip(forward),reverse),axis=0)


    #format_txt = '%.' + str(n_sig_fig) + 'f'
    #np.savetxt("harmtestpost.csv", ml_reshaped[::-1,:], delimiter=",", fmt=format_txt)

    #binarydata = ml_reshaped[::-1,:].tobytes()      # This weird correction is here so that time series is in the right order
    #print(time.time() - tim3)

    return ml_reshaped

def genericoutpufile(NN,filters,modeltype,freqused,activtype):
    # make a file for reaction mechanisms  but check one isnt there to begin with
    file = True
    i = 0
    while file:
        try:
            if i == 0:
                filename = "Output_" + NN +freqused +"Hz" +modeltype+activtype+ filters +"_"+ str(datetime.date.today())
                os.makedirs(filename)
            else:
                filename = "Output_" + str(i) + "_" + NN +freqused +"Hz" +modeltype+activtype+ filters + "_" + str(datetime.date.today())
                os.makedirs(filename)
            file = False
        except:
            print("file already exists")
        finally:
            i += 1

    return filename

#summary for the deepest NN model
def deepmodel(imagedimensions,spatialdropout,dnnHLNodeNumber,n_labels,filterdic,activtype):
    N = 32
    filtersize1 = (1, 1)
    print(filterdic)

    # sets data type lower for speed and memory storage
    policy = mixed_precision.experimental.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)

    input_shape = Input(shape=(imagedimensions[0], imagedimensions[1], 1))

    inital_layer = Conv2D(N, filtersize1, activation="relu", padding="same")(input_shape)
    inital_layer = Conv2D(N, filtersize1, activation='relu', padding="same")(inital_layer)
    inital_layer = SpatialDropout2D(spatialdropout)(inital_layer)

    # low res sweep

    towerlowres = Conv2D(N, filterdic[0], activation=activtype, padding="same")(inital_layer)
    towerlowres = Conv2D(N, filterdic[0], activation=activtype, padding="same")(towerlowres)
    towerlowres = SpatialDropout2D(spatialdropout)(towerlowres)
    towerlowres = MaxPooling2D(pool_size=(2, 2))(towerlowres)

    towerlowres = Conv2D(N, filterdic[1], activation=activtype, padding="same")(towerlowres)
    towerlowres = Conv2D(N, filterdic[1], activation=activtype, padding="same")(towerlowres)
    towerlowres = SpatialDropout2D(spatialdropout)(towerlowres)
    towerlowres = MaxPooling2D(pool_size=(2, 2))(towerlowres)

    towerlowres = Conv2D(2 * N, filterdic[2], activation=activtype, padding="same")(towerlowres)
    towerlowres = Conv2D(2 * N, filterdic[2], activation=activtype, padding="same")(towerlowres)
    towerlowres = SpatialDropout2D(spatialdropout)(towerlowres)
    towerlowres = MaxPooling2D(pool_size=(2, 2))(towerlowres)

    towerlowres = Conv2D(2 * N, filterdic[3], activation=activtype, padding="same")(towerlowres)
    towerlowres = Conv2D(2 * N, filterdic[3], activation=activtype, padding="same")(towerlowres)
    towerlowres = SpatialDropout2D(spatialdropout)(towerlowres)
    towerlowres = MaxPooling2D(pool_size=(2, 2))(towerlowres)

    # flatten to 1D data
    merged = Flatten()(towerlowres)
    # dense connection to a hidden layer of variable number of nodes
    merged = Dense(dnnHLNodeNumber, dtype='float32', activation="relu")(merged)

    # final output layer of the 3 classifications (E, EE and EC mechanisms)
    output = Dense(n_labels, dtype='float32',activation='softmax')(merged)
    # sigmoid activation function for the final classification layer
    # compile the model using the ADAM optimizer for a specified learning rate

    model = keras.Model(input_shape, output)


    return model

#summary for the deepest NN model
def widemodel(imagedimensions,spatialdropout,dnnHLNodeNumber,n_labels,filterdic,activtype):
    N = 32

    # sets data type lower for speed and memory storage

    input_shape = Input(shape=(imagedimensions[0], imagedimensions[1], 1))

    filtersize1 = (1,1)

    policy = mixed_precision.experimental.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)

    inital_layer = Conv2D(N, filtersize1, activation='relu', padding="same")(input_shape)
    inital_layer = Conv2D(N, filtersize1, activation='relu', padding="same")(inital_layer)
    inital_layer = SpatialDropout2D(spatialdropout)(inital_layer)
    inital_layer = MaxPooling2D(pool_size=(2, 2))(inital_layer)

    # low res sweep

    towerlowres = Conv2D(N, filterdic[0], activation=activtype, padding="same")(inital_layer)
    towerlowres = Conv2D(N, filterdic[0], activation=activtype, padding="same")(towerlowres)
    towerlowres = SpatialDropout2D(spatialdropout)(towerlowres)
    towerlowres = MaxPooling2D(pool_size=(2, 2))(towerlowres)

    towerlowres = Conv2D(1 * N, filterdic[1], activation=activtype, padding="same")(towerlowres)
    towerlowres = Conv2D(1 * N, filterdic[1], activation=activtype, padding="same")(towerlowres)
    towerlowres = SpatialDropout2D(spatialdropout)(towerlowres)
    towerlowres = MaxPooling2D(pool_size=(2, 2))(towerlowres)

    towerlowres = Conv2D(2 * N, filterdic[2], activation=activtype, padding="same")(towerlowres)
    towerlowres = Conv2D(2 * N, filterdic[2], activation=activtype, padding="same")(towerlowres)
    towerlowres = SpatialDropout2D(spatialdropout)(towerlowres)
    towerlowres = MaxPooling2D(pool_size=(2, 2))(towerlowres)

    # med res
    towerhighres = Conv2D(2 * N, filterdic[3], activation=activtype, padding="same")(inital_layer)
    towerhighres = Conv2D(2 * N, filterdic[3], activation=activtype, padding="same")(towerhighres)
    towerhighres = SpatialDropout2D(spatialdropout)(towerhighres)
    towerhighres = MaxPooling2D(pool_size=(8, 8))(towerhighres)

    print(towerlowres.shape)
    print(towerhighres.shape)
    merged = concatenate([towerlowres, towerhighres], axis=3)
    print(merged.shape)
    # 2nd fully connected layer
    merged = Flatten()(merged)
    # dense connection to a hidden layer of variable number of nodes
    merged = Dense(dnnHLNodeNumber, dtype='float32', activation="relu")(merged)

    # final output layer of the 3 classifications (E, EE and EC mechanisms)
    output = Dense(n_labels, dtype='float32', activation='softmax')(merged)
    # sigmoid activation function for the final classification layer


    model = keras.Model(input_shape, output)


    return model

def accuracyplot(history,harmnum,filename):
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(filename + "/accuracyplot_H" + str(harmnum) + ".png")

    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(filename + "/Lossplot_H" + str(harmnum) + ".png")

    return

def genericfitingdata(filename, harmnum, intertic,model_numcode,models,harmtrain_mech ):
    # prints some text houtput
    f = open(filename + "/generic_fitting_data.txt", "a")
    f.write("Harmonic " + str(harmnum) + " harmdata\n")
    f.write("Completion time of iteration (min): " + str((time.time() - intertic) / 60) + "\n")

    f.write("Classification order")
    for key, value in model_numcode.items():
        f.write(key + " : " + str(value) + "\n")
    f.write("\n")

    f.write("iteration model split number per mech: \n")
    k = 0
    for mechs in models:
        f.write(mechs + ': ')
        k += harmtrain_mech.count(mechs)
        f.write(str(int(harmtrain_mech.count(mechs))) + "\t")
    f.write('Total: ' + str(k))
    f.write("\n\n")
    f.close()

    return