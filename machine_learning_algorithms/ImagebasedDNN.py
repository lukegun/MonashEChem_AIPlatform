import numpy as np
import tensorflow as tf
from tensorflow import keras
import random
from random import shuffle # mixing up or currently ordered data that might lead our network astray in training.
import os
import datetime
import matplotlib.pyplot as plt
import time
import psycopg2
from multiprocessing import Pool # for collecting the data

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Lambda
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv1D, Activation , BatchNormalization
from tensorflow.keras.layers import AveragePooling2D, SpatialDropout2D
from tensorflow.keras.layers import Input, concatenate, Add # this is for layered inputs
from tensorflow.keras import mixed_precision


class Machine_learning_class(keras.utils.Sequence):
    # initialisation
    def __init__(self, currentdata, mechaccept, potdic, harmnum, batchsize=32, dim=(100, 100), serverdata=["testdb"], n_classes=1000,
                 n_channels=1, model_numcode={"E": 0}, cpu_number=1, exp_data={}, modeltype="deep", noise = 0):
        self.harmnum = harmnum  # which number harmonic we are identifying
        self.batch_size = batchsize  # numer of data points we are taking
        self.imagedimensions = dim
        self.serverdata = serverdata
        inputdata = NN_codegenerator(mechaccept, model_numcode)
        self.batch_indexes = NN_batchsetter(inputdata, batchsize)  # Generate indexes of the batch
        self.indices = len(inputdata)
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.cpu_number = cpu_number
        self.exp_data = exp_data
        self.modeltype = modeltype
        self.noise = noise
        if self.noise == 0:
            self.noiseproc = False
        else:
            self.noiseproc = True

        self.diccurrent = {}
        for j in range(len(mechaccept)):
            self.diccurrent.update({mechaccept[j][0]:currentdata[j]})

        self.potdic = potdic


    # THIS OVERWRITE THINGS IN TH KERAS SEQUENCE CLASS
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        t1 = time.time()
        list_IDs_temp = self.batch_indexes[index]

        # Generate data
        X, y = self.FTACV_CNN_generator(list_IDs_temp)
        print(" Batch Time: " + str(time.time() - t1))
        # dataset = self.FTACV_CNN_generator(list_IDs_temp)

        # print(np.random.randint(0,100000))
        # print(dataset)
        # print(y)

        return np.array(X), np.array(y)

    def __len__(self):
        # Denotes the number of batches per epoch
        return self.indices // self.batch_size

    # Generator for the machine learning model
    def FTACV_CNN_generator(self, batchdata):  # harmnum is the harmonic we are training for

        # Initialization
        X = np.empty((self.batch_size, self.imagedimensions[1], self.imagedimensions[0], self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # with Pool(processes=Ncore) as p:

        # FIT = p.map(run, listin)
        react_ID = []
        for i, inputs in enumerate(batchdata):
            #dictionary get

            react_ID.append(inputs[0])
            ##image = self.image_collector(react_ID) # inputs[1] = Classification
            # X[i,:,:,:] = image[0,:,:,:]
            y[i] = inputs[1]

        run = self.image_collector

        with Pool(processes=self.cpu_number) as p:

            Batchimage = p.map(run, react_ID)

        for i in range(self.batch_size):
            X[i, :, :, :] = Batchimage[i][0, :, :, :]

        # Convert to  a tf dataset
        # dataset = tf.data.Dataset.from_tensor_slices((X, y))

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
    def image_collector(self, react_ID):


        cuurr = self.diccurrent.get(react_ID)
        pot = self.potdic.get(react_ID)

        minI = -max(abs(cuurr))
        maxI = max(abs(cuurr))

        # harmonic_data = np.frombuffer(cuurr[0][self.harmnum]) .reshape(self.imagedimensions[0], self.imagedimensions[1])
        """print("WARNING: DC works but is cheap and assumes that Estart EEnd ar all same")"""
        DC_datasweep1 = NN_ILgenerator_dc_exp(cuurr[:int(len(cuurr))], self.imagedimensions[0], self.imagedimensions[1],minI,maxI,pot)


        DC_datasweep = DC_datasweep1.reshape(1, self.imagedimensions[1], self.imagedimensions[0], 1)

        return DC_datasweep

def NN_codegenerator(inputdata, model_numcode):
    x = []
    for stuff in inputdata:
        stuff[1] = model_numcode.get(stuff[1])
        x.append(stuff)
    return x

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

#summary for the deepest NN model
def garethmodel(imagedimensions,dnnHLNodeNumber,n_labels):
    N = 32

    # sets data type lower for speed and memory storage

    input_shape = Input(shape=(imagedimensions[0], imagedimensions[1], 1))

    filtersize3 = (3,3)

    policy = mixed_precision.experimental.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)

    inital_layer = Conv2D(N, filtersize3, activation='relu')(input_shape)
    inital_layer = MaxPooling2D(pool_size=(2, 2))(inital_layer)
    inital_layer = Conv2D(N, filtersize3, activation='relu')(inital_layer)
    inital_layer = MaxPooling2D(pool_size=(2, 2))(inital_layer)

    inital_layer = Conv2D(2*N, filtersize3, activation='relu')(inital_layer)
    inital_layer = MaxPooling2D(pool_size=(2, 2))(inital_layer)

    # 2nd fully connected layer
    merged = Flatten()(inital_layer)
    # dense connection to a hidden layer of variable number of nodes
    merged = Dense(dnnHLNodeNumber, dtype='float32', activation="relu")(merged)

    merged = Dropout(0.5)(merged)

    # final output layer of the 3 classifications (E, EE and EC mechanisms)
    output = Dense(n_labels, dtype='float32', activation='softmax')(merged)
    # sigmoid activation function for the final classification layer

    model = keras.Model(input_shape, output)

    return model

def NN_ILgenerator_dc_exp(current,Nx,Ny,minI,maxI,pot):
    n_i = int(Ny)
    n_e = int(Nx)
    n_sig_fig = 4

    Estart = pot[0]
    Erev = pot[1]
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

def genericoutpufile(NN,modeltype):
    # make a file for reaction mechanisms  but check one isnt there to begin with
    file = True
    i = 0
    while file:
        try:
            if i == 0:
                filename = "Output_" + NN +modeltype +"_"+ str(datetime.date.today())
                os.makedirs(filename)
            else:
                filename = "Output_" + str(i) + "_" + NN +modeltype + "_" + str(datetime.date.today())
                os.makedirs(filename)
            file = False
        except:
            print("file already exists")
        finally:
            i += 1

    return filename

# gets the potential range from the sql data
def potentialgetter(serverdata,mechaccept):

    try:
        connection = psycopg2.connect(user=serverdata[0],
                                      password=serverdata[1],
                                      host=serverdata[2],
                                      port=serverdata[3],
                                      database=serverdata[4])

        cursor = connection.cursor()

        # breaks all of this up so that there isn't like 2 times the database present at one time
        list = []
        i = 1
        for ID in mechaccept:
            list.append(ID[0])
            i += 1

        listtot = tuple(list)

        tots = []
        mechaccept = []
        j = 0

        t1 = time.time()
        yaya = """SELECT "Reaction_ID","Estart","Eend" FROM "Simulatedparameters" WHERE "Reaction_ID" IN %s"""
        cursor.execute(yaya, (listtot,))
        xtotbigtot = cursor.fetchall()
        print(time.time() - t1)

        potdictest = {}

        for ID in list: # itterate and search for arrays allocated in above section for use
            for i in range(len(list)):
                if ID == xtotbigtot[i][0]:
                    potdictest.update({ID:[xtotbigtot[i][1],xtotbigtot[i][2]]})
                    break


    except (Exception, psycopg2.Error) as error:
        print("error in potgetter,", error)
    finally:
        # This is needed for the
        if (connection):
            cursor.close()
            connection.close()

    # insert the values into sql

    return potdictest