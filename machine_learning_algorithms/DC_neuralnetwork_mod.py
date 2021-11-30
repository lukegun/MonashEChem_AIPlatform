

import psycopg2
import numpy as np
import random
from random import shuffle # mixing up or currently ordered data that might lead our network astray in training.
import os
import datetime
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool # for collecting the data
import tensorflow as tf
from tensorflow import keras


class Machine_learning_class(keras.utils.Sequence):
    # initialisation
    def __init__(self, inputdata, harmnum, batchsize=32, dim=(100, 100), serverdata=["testdb"], n_classes=1000,
                 n_channels=1, model_numcode={"E": 0}, cpu_number=1, exp_data={}, modeltype="deep", noise = 0):
        self.harmnum = harmnum  # which number harmonic we are identifying
        self.batch_size = batchsize  # numer of data points we are taking
        self.imagedimensions = dim
        self.serverdata = serverdata
        inputdata = NN_codegenerator(inputdata, model_numcode)
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
            # choose random index in features
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

        connection = psycopg2.connect(user="postgres",
                                      password="Quantum2",
                                      host="localhost",
                                      port="5432",
                                      database=self.serverdata[0])

        cursor = connection.cursor()

        # react_class_tot = []
        # react_class = []
        # put in the parameers where they sould go

        try:
            # print(react_ID)
            # gets the EXPsetrow for reaction mechanism and experimental reaction mech
            tab = """SELECT "TotalCurrent" FROM "HarmTab" WHERE "Reaction_ID" =  %s"""
            cursor.execute(tab, (react_ID,))
            # extracts the
            cuurr = np.array(cursor.fetchall()[0][0])

            # format_txt = '%.' + str(4) + 'f' this is for saving the dta to a cs

            if self.noiseproc:
                print("Using noise")
                Nc = len(cuurr)
                std = self.noise * max(abs(cuurr))
                cuurr = cuurr[:] + np.random.normal(0, std, Nc)

            # needs to be recalculculated after due to chance of being larger then image
            minI = -max(abs(cuurr))
            maxI = max(abs(cuurr))

            # harmonic_data = np.frombuffer(cuurr[0][self.harmnum]) .reshape(self.imagedimensions[0], self.imagedimensions[1])
            """print("WARNING: DC works but is cheap and assumes that Estart EEnd ar all same")"""
            DC_datasweep1 = NN_ILgenerator_dc_exp(cuurr[:int(len(cuurr)/4)], self.imagedimensions[0], self.imagedimensions[1],minI,maxI)
            """plt.figure(1)

            Estart = 0.67
            Erev = -0.67
            Ncycles = 1
            dpoints = len(cuurr[:int(len(cuurr)/4)])

            for i in range(int(Ncycles)):
                DC1 = np.linspace(Estart, Erev, int(dpoints / (2 * Ncycles)))
                DC2 = np.linspace(Erev, Estart, int(dpoints / (2 * Ncycles)))
                if i == 0:
                    DC = np.append(DC1, DC2)
                else:
                    x = np.append(DC1, DC2)
                    DC = np.append(DC, x)

            e_dc = DC

            if react_ID < 120:
                plt.plot(e_dc,cuurr[:int(len(cuurr)/4)]/max(abs(cuurr[:int(len(cuurr)/4)])),color = 'b')
            else:
                plt.plot(e_dc,cuurr[:int(len(cuurr) / 4)]/max(abs(cuurr[:int(len(cuurr)/4)])),color = 'r')
            plt.savefig("imge2.png")
            time.sleep(0.5)"""

            DC_datasweep1 = DC_datasweep1.reshape(1, self.imagedimensions[1], self.imagedimensions[0], 1)

            """
                        Required to get into a format that is easy for Keras (and Tensorflow underneath) to use. Format is:

                        (n_images, IMAGE_X, IMAGE_Y, n_channels)

                        where n_images is the number of images to stack on top of each other in a single batch or the entire training 
                        set, the image has resolution X,Y and the number of channels is 1 for black and white or 3 for RGB colour images.            
                        """

            if self.modeltype != "gareth":

                DC_datasweep = DC_datasweep1

                for i in range(self.n_channels- 1):

                    DC_datasweep2 = NN_ILgenerator_dc_exp(cuurr[(i+1)*int(len(cuurr) / 4):(i+2)*int(len(cuurr) / 4)], self.imagedimensions[0],
                                                self.imagedimensions[1],minI,maxI)

                    DC_datasweep2 = DC_datasweep2.reshape(1, self.imagedimensions[1], self.imagedimensions[0], 1)

                    DC_datasweep = np.concatenate((DC_datasweep,DC_datasweep2),axis = 3)

            else:
                DC_datasweep = DC_datasweep1

        except (Exception, psycopg2.Error) as error:
            print("error message from sql_updater3,", error)
            exit()

        if (connection):
            cursor.close()
            connection.close()

        return DC_datasweep

# connect to server and get an list of all reaction IDs and paired with class of form react_class = [[ReactionID,Reactionmech]]
def image_gen_4N(cuurr,imagedimensions,modeltype,n_channels):

    minI = -max(abs(cuurr))
    maxI = max(abs(cuurr))

    DC_datasweep1 = NN_ILgenerator_dc_exp(cuurr[:int(len(cuurr)/4)], imagedimensions[0], imagedimensions[1],minI,maxI)


    DC_datasweep1 = DC_datasweep1.reshape(1, imagedimensions[1], imagedimensions[0], 1)

    """
                        Required to get into a format that is easy for Keras (and Tensorflow underneath) to use. Format is:

                        (n_images, IMAGE_X, IMAGE_Y, n_channels)

                        where n_images is the number of images to stack on top of each other in a single batch or the entire training 
                        set, the image has resolution X,Y and the number of channels is 1 for black and white or 3 for RGB colour images.            
    """

    if modeltype != "gareth":

        DC_datasweep = DC_datasweep1

        for i in range(n_channels- 1):

            DC_datasweep2 = NN_ILgenerator_dc_exp(cuurr[(i+1)*int(len(cuurr) / 4):(i+2)*int(len(cuurr) / 4)], imagedimensions[0],
                                                imagedimensions[1],minI,maxI)

            DC_datasweep2 = DC_datasweep2.reshape(1, imagedimensions[1], imagedimensions[0], 1)
            DC_datasweep = np.concatenate((DC_datasweep,DC_datasweep2),axis = 3)

    else:
        DC_datasweep = DC_datasweep1

    return DC_datasweep

def DC_NN_setter(datain,harmnum):
    data_ID = []
    data_mech = []
    data = []
    print(len(datain))
    for inputs in datain:

        data_ID.append(inputs[0])
        data_mech.append(inputs[1])
        data.append(
                [inputs[0], inputs[1]])  # required as i don't trust other methods to be stable on parallelelisation
    return data, data_mech, data_ID

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


def NN_codegenerator(inputdata, model_numcode):

    x = []
    for stuff in inputdata:
        stuff[1] = model_numcode.get(stuff[1])
        x.append(stuff)
    return x

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

#summary for the deepest NN model
def garethmodel4N(imagedimensions,dnnHLNodeNumber,n_labels):
    N = 32

    # sets data type lower for speed and memory storage

    input_shape = Input(shape=(imagedimensions[0], imagedimensions[1], 4))

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

#summary for the deepest NN model
def garethmodel3N(imagedimensions,dnnHLNodeNumber,n_labels):
    N = 32

    # sets data type lower for speed and memory storage

    input_shape = Input(shape=(imagedimensions[0], imagedimensions[1], 3))

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


#summary for the deepest NN model
def deepmodel_DC(imagedimensions,spatialdropout,dnnHLNodeNumber,n_labels,filterdic,activtype,channel):
    N = 32*2
    filtersize1 = (1, 1)
    filtersize3 = (3, 3)
    filtersize5 = (5, 5)
    filtersize7 = (7, 7)
    filtersize9 = (9, 9)
    filtersize11 = (11, 11)
    filtersize25 = (25, 25)

    spatialdropout = 0.5
    dropout = 0.5

    # sets data type lower for speed and memory storage
    policy = mixed_precision.experimental.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)

    input_shape = Input(shape=(imagedimensions[0], imagedimensions[1], channel))

    #com = Conv2D(int(N/2), filtersize25, activation='relu', padding="same")(input_shape)
    com = Conv2D(N, filtersize25, activation='relu', padding="same")(input_shape)
    com = MaxPooling2D(pool_size=(2, 2))(com)
    #com2 = Conv2D(int(N/2), filtersize11, activation='relu', padding="same")(input_shape[:, :, :, 0:1])
    #com2 = MaxPooling2D(pool_size=(2, 2))(com2)
    #com = concatenate([com, com2], axis=3)
    com = Conv2D(N, filtersize11, activation='relu', padding="same")(com)
    com = SpatialDropout2D(spatialdropout)(com)
    com = MaxPooling2D(pool_size=(2, 2))(com)
    com = Conv2D(2*N, filtersize7, activation='relu', padding="same")(com)
    com = Conv2D(2 * N, filtersize7, activation='relu', padding="same")(com)
    com = SpatialDropout2D(spatialdropout)(com)
    joined = MaxPooling2D(pool_size=(2, 2))(com)

    #joined = concatenate([inter,com],axis = 3)
    joined = Conv2D(4*N, filtersize5, activation='relu',padding="same")(joined)
    joined = SpatialDropout2D(spatialdropout)(joined)
    joined = MaxPooling2D(pool_size=(2, 2))(joined)
    joined = Conv2D(4*N, filtersize5, activation='relu', padding="same")(joined)
    joined = SpatialDropout2D(spatialdropout)(joined)
    joined = MaxPooling2D(pool_size=(2,2))(joined)


    # flatten to 1D data
    merged = Flatten()(joined)
    # dense connection to a hidden layer of variable number of nodes
    merged = Dense(dnnHLNodeNumber, dtype='float32', activation="relu")(merged)
    merged = Dropout(dropout)(merged)
    merged = Dense(dnnHLNodeNumber, dtype='float32', activation="relu")(merged)
    #merged = Dropout(dropout)(merged)
    merged = Dense(dnnHLNodeNumber, dtype='float32', activation="relu")(merged)
    #merged = Dropout(dropout)(merged)
    # final output layer of the 3 classifications (E, EE and EC mechanisms)
    output = Dense(n_labels, dtype='float32',activation='softmax')(merged)
    # sigmoid activation function for the final classification layer
    # compile the model using the ADAM optimizer for a specified learning rate

    model = keras.Model(input_shape, output)

    return model

#summary for the deepest NN model
def OGdeepmodel_DC(imagedimensions,spatialdropout,dnnHLNodeNumber,n_labels,filterdic,activtype):
    N = 32
    filtersize1 = (1, 1)
    filtersize3 = (3, 3)
    print(filterdic)

    # sets data type lower for speed and memory storage
    policy = mixed_precision.experimental.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)

    input_shape1 = Input(shape=(imagedimensions[0], imagedimensions[1], 1))


    inital_layer = Conv2D(N, filtersize3, activation="relu", padding="same")(input_shape1)
    #inital_layer = Conv2D(N, filtersize3, activation='relu', padding="same")(inital_layer)
    inital_layer = SpatialDropout2D(spatialdropout)(inital_layer)
    inital_layer = MaxPooling2D(pool_size=(2, 2))(inital_layer)

    # low res sweep

    towerlowres = Conv2D(N, filterdic[0], activation=activtype, padding="same")(inital_layer)
    #towerlowres = Conv2D(N, filterdic[0], activation=activtype, padding="same")(towerlowres)
    towerlowres = SpatialDropout2D(spatialdropout)(towerlowres)
    towerlowres = MaxPooling2D(pool_size=(2, 2))(towerlowres)

    towerlowres = Conv2D(N, filterdic[1], activation=activtype, padding="same")(towerlowres)
    #towerlowres = Conv2D(N, filterdic[1], activation=activtype, padding="same")(towerlowres)
    towerlowres = SpatialDropout2D(spatialdropout)(towerlowres)
    main = MaxPooling2D(pool_size=(2, 2))(towerlowres)


    side = Conv2D(2 * N, filterdic[2], activation=activtype, padding="same")(input_shape)
    #side = Conv2D(2 * N, filterdic[2], activation=activtype, padding="same")(side)
    side = SpatialDropout2D(spatialdropout)(side)
    side = MaxPooling2D(pool_size=(8, 8))(side)

    towerlowres = concatenate([main, side], axis=3)

    towerlowres = Conv2D(4 * N, filterdic[3], activation=activtype, padding="same")(towerlowres)
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