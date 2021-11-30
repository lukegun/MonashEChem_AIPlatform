import pandas as pd

import DNNtimecluster_mod as TC_mod
import TrainDNNtimeclass_mod as TDNN
import DC_neuralnetwork_mod as NNt
import ImagebasedDNN as IB_NNt
import time
import sys
import numpy as np
"""iMPORT DNN settings""" #Need to get which experimental parameters where loading
time1 = time.time()
input_name = sys.argv[1]

serverdata, supivisorclass, ACCase, variate, DNNmodel, cpu_workers, gpu_workers, deci, harmdata, modelparamterfile, \
    trainratio, reactionmech = TDNN.inputloader(input_name)


# Something to classify the DNN label and data to a specific model
print(serverdata)
# THIS GETS THE LABELS

if supivisorclass == "reactmech":  # classifiers the the DNN supivised to reaction mechs
    # """GET EXPsetrow THAT RELATE TO THE ABOVE AC SINE PROPERTIES AND  MODEL""" #exp = [[expsetrow,reactionmech],...]

    exp_class = TDNN.EXPsetrow_collector(serverdata, reactionmech)

    exp_data = TDNN.EXPsetrow_dcdata(serverdata, exp_class)

        # create list of form react_class = [[[ReactionID,Reactionmech]]] for each reac mech
    react_class = TDNN.ReactID_collector(serverdata, exp_class)

        # count number for each reaction model and check if same
    Narray = TDNN.Narray_count(react_class)

elif supivisorclass == "clustering":

    exp_class = TDNN.EXPsetrow_collector(serverdata, reactionmech)

    exp_data = TDNN.EXPsetrow_dcdata(serverdata, exp_class)

    # create list of form react_class = [[[ReactionID,Reactionmech]]] for each reac mech
    react_class = TDNN.ReactID_collector(serverdata, exp_class)

    #"""something to get the AC clustered label"""
    if ACCase:
        from ACgrouping import AC_clustergroup

        # sets up the preclustering labels
        preclusterreactionmech = reactionmech
        preclusterreact_class = react_class

        print("clustering stuff")
        # SOME GROUPING LABEL IN HERE
        Groupclass = TDNN.ACreactclusterupdate(react_class, serverdata,num=25) # number is cut of for subgroups
        """something to get reactionmech labels for AC CASE"""
        cuttoff = 1000  # groups need to be larger then this number

        groupeddic = AC_clustergroup(Groupclass, cuttoff)
        groupeddicnew = {}
        for labels,lists in groupeddic.items():
            i = 1
            newlist = []
            s = "H0" + labels[:2]
            for i in range(1, int(len(labels) / 2)):
                s += "_H" + str(i) + labels[int(i * 2):int(i * 2) + 2]
            newlabel = s


            for values in lists:
                s = "H0" + values[:2]
                print(values)
                for i in range(1,int(len(values)/2)):
                    s += "_H" + str(i) + values[int(i*2):int(i*2)+2]
                print(s)

                # creates to proper sql format
                if i != 8:
                    for j in range(i + 1, 9):
                        s += "_H" + str(j) + "00"

                newlist.append(s)
            groupeddicnew.update({newlabel:newlist})
        groupeddic = groupeddicnew

        # extracts clustered reactiong mechs in sql freadly manner
        x = []
        for val in groupeddic:
            x.append(val)
        reactionmech = x

        # convert reaction labels to clustered labels
        react_class = TDNN.ACreactclusterupdate2(serverdata,groupeddic)


    else: # """something to get the DC clustered label"""

        react_class, reactionmech = TDNN.DCreactclusterupdate(react_class,serverdata)

        """something to get reactionmech labels"""

    Narray = TDNN.Narray_count(react_class)

 # train data is for train NN, testdata is for testing NN
testdata, traindata, Ntest, Ntrain = TC_mod.suffle_splittrainratio(trainratio, react_class)

modeldic = TDNN.modeldataloader(modelparamterfile)
"""May need something here to import model specific parpameters"""
"""Load the reactmech or classes from the sql libary where exp setting is X"""

# sets up model numbers
model_numcode = {}
for i in range(len(reactionmech)):
    model_numcode.update({reactionmech[i]:i})
print( model_numcode)
Nmodels = len(reactionmech)

#seperates the input data into each model type
if variate: #use the multivariate models TRUE = multivariate

    print("Running classifier on AC data")

    harmtrain, harmtrain_mech, harmtrain_ID = NNt.DC_NN_setter(traindata, harmdata)  # training data
    print("worked")
    harmtest, harmtest_mech, harmtest_ID = NNt.DC_NN_setter(testdata, harmdata)  # testing data for validation
    print("worked2")

    # small change made fater less likely to parralellise it
    currentdatatest, mechaccepttest = TDNN.ACsqlcurrentcollector(serverdata, harmdata, harmtest, deci,DNNmodel)
    print("print3")
    currentdatatrain, mechaccepttrain = TDNN.ACsqlcurrentcollector(serverdata, harmdata, harmtrain, deci,DNNmodel)

    #load all the data from sql databaase and save to RA

    print("Poo")
    x = []
    for values in mechaccepttest:
        x.append(model_numcode.get(values[1]))
    mechlabelstest = x

    x = []
    for values in mechaccepttrain:
        x.append(model_numcode.get(values[1]))
    mechlabelstrain = x

    # it can then be passed to the GPU in Batches
    if DNNmodel == "rocket": # use the DNN rocket model
        # RAM ISSUES WITH THIS ONE DOESN"T WORK WELL ON 15k data at 80% train due to classifier
        print("Running rocket classifier on AC data")
        from sktime.transformations.panel.rocket import Rocket
        from sklearn.linear_model import RidgeClassifierCV
        from sktime.datasets import load_arrow_head  # univariate dataset

        # from Rocketmodfunctions import train as Rtrain

        # change stuff to pd series where row is input, column is dimension and "3rd dim" is current trace for compadability
        currentdatatrain = pd.DataFrame(currentdatatrain)
        currentdatatest = pd.DataFrame(currentdatatest)
        print("done")
        print(len(currentdatatrain))
        print(len(currentdatatest))
        # these are 1D so not required
        mechlabelstrain = np.array(mechlabelstrain)

        #mechlabelstest = np.array(mechlabelstest)

        rocket = Rocket(num_kernels=1000)  # by default, ROCKET uses 10,000 kernels ideally but where trying to get ram working
        rocket.fit(currentdatatrain)
        X_train_transform = rocket.transform(currentdatatrain)

        print("Data Trasnsformed fitting DNN")

        classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
        classifier.fit(X_train_transform, mechlabelstrain)

        print("Completion Time (mins): " + str((time.time() - time1) / 60))

        print("Data classified checking labeling")
        # test classification accuracy
        X_test_transform = rocket.transform(currentdatatest)
        print(classifier.score(X_test_transform, mechlabelstest))

    elif DNNmodel == "inceptiontime": # use the inception time DNN model
        print("Running inceptiontime classifier on DC data")
        from classifiers.inception import Classifier_INCEPTION
        from tensorflow import keras

        Nchanel = len(harmdata)
        # from classifiers.nne import Classifier_NNE

        # change stuff to pd series where row is input, column is dimension and "3rd dim" is current trace for compadability
        #x = np.empty((len(currentdatatrain),deci,len(harmdata)))
        #i = 0
        #for data in currentdatatrain:
        #    print(data.shape)
        #    x[i,:,:] = data[:,:]
        #currentdatatrain = x
        currentdatatrain = np.array(currentdatatrain)
        currentdatatest = np.array(currentdatatest)

        # these are 1D so not required
        mechlabelstrain = np.array(mechlabelstrain)
        mechlabelstrain = keras.utils.to_categorical(mechlabelstrain, num_classes=Nmodels)
        mechlabelstest1 = np.array(mechlabelstest)
        mechlabelstest = keras.utils.to_categorical(mechlabelstest, num_classes=Nmodels)

        print("Data Collected fitting DNN")

        # model = Classifier_INCEPTION("output",None,nb_classes=Nmodels )
        model = Classifier_INCEPTION("output", (deci, Nchanel), nb_classes=Nmodels, verbose=True,
                                     nb_epochs=3)  # using 20 epochs for ease
        model.build_model((deci, Nchanel), Nmodels)
        model.fit(currentdatatrain, mechlabelstrain, currentdatatest, mechlabelstest, mechlabelstest1)
        print("TestingDNN")
        """# NEED SOMETHING HERE TO TEST THE MODEL ACCURACY

            # also need to clean up the output metrics"""


    else:
        print("ERROR: No DNN model has been classified")
        exit()

else: # use the univarate models Mainly used in DC Case

    harmdata = [-1]     # just set it up to use the DC case
    harmnum = -1

    print("change these for specific data")
    harmtrain, harmtrain_mech, harmtrain_ID = NNt.DC_NN_setter(traindata, harmnum)  # training data
    harmtest, harmtest_mech, harmtest_ID = NNt.DC_NN_setter(testdata, harmnum)  # testing data for validation

    currentdatatest, mechaccepttest = TDNN.sqlcurrentcollector(serverdata, harmnum, harmtest, deci)
    currentdatatrain, mechaccepttrain = TDNN.sqlcurrentcollector(serverdata, harmnum, harmtrain, deci)
    #above is (current data, [reaction_ID, Reactionmech])

    x = []
    for values in mechaccepttest:
        x.append(model_numcode.get(values[1]))
    mechlabelstest = x

    x = []
    for values in mechaccepttrain:
        x.append(model_numcode.get(values[1]))
    mechlabelstrain = x

    # load all the data from sql databaase and save to RAM

    # it can then be passed to the GPU in Batches

    if DNNmodel == "rocket": # use the DNN rocket model
        # RAM ISSUES WITH THIS ONE DOESN"T WORK WELL ON 15k data at 80% train due to classifier
        print("Running rocket classifier on DC data")
        from sktime.transformations.panel.rocket import Rocket
        from sklearn.linear_model import RidgeClassifierCV, LogisticRegression
        from sktime.datasets import load_arrow_head  # univariate dataset
        #from Rocketmodfunctions import train as Rtrain

        # change stuff to pd series where row is input, column is dimension and "3rd dim" is current trace for compadability
        currentdatatrain = pd.DataFrame(pd.Series(currentdatatrain))
        currentdatatest = pd.DataFrame(pd.Series(currentdatatest))

        # these are 1D so not required
        mechlabelstrain = np.array(mechlabelstrain)
        mechlabelstest = np.array(mechlabelstest)

        rocket = Rocket(num_kernels=1000)  # by default, ROCKET uses 10,000 kernels ideally but where trying to get ram working
        rocket.fit(currentdatatrain)
        X_train_transform = rocket.transform(currentdatatrain)

        print("Data Trasnsformed fitting DNN")

        classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
        classifier.fit(X_train_transform, mechlabelstrain)

        print("Data classified checking labeling")
        # test classification accuracy
        X_test_transform = rocket.transform(currentdatatest)
        print(classifier.score(X_test_transform, mechlabelstest))

        #print(classifier.score(X_test_transform, mechlabelstest))

    elif DNNmodel == "inceptiontime": # use the inception time DNN model
        print("Running inceptiontime classifier on DC data")
        from classifiers.inception import Classifier_INCEPTION
        from tensorflow import keras
        Nchanel = 1
        #from classifiers.nne import Classifier_NNE

        # change stuff to pd series where row is input, column is dimension and "3rd dim" is current trace for compadability
        currentdatatrain = np.array(currentdatatrain)
        currentdatatest = np.array(currentdatatest)
        ntrain,r = currentdatatrain.shape
        currentdatatrain = currentdatatrain.reshape(ntrain,r,Nchanel)


        ntest, r = currentdatatest.shape
        currentdatatest = currentdatatest.reshape(ntest, r, Nchanel)

        # these are 1D so not required
        mechlabelstrain = np.array(mechlabelstrain)
        mechlabelstrain = keras.utils.to_categorical(mechlabelstrain, num_classes=Nmodels)
        mechlabelstest1 = np.array(mechlabelstest)
        mechlabelstest = keras.utils.to_categorical(mechlabelstest1, num_classes=Nmodels)

        print("Data Collected fitting DNN")

        #model = Classifier_INCEPTION("output",None,nb_classes=Nmodels )
        print()
        model = Classifier_INCEPTION("output", (deci,Nchanel), nb_classes=Nmodels,verbose=True,nb_epochs=3) # using 20 epochs for ease
        model.build_model((deci,Nchanel),Nmodels)
        model.fit(currentdatatrain, mechlabelstrain,currentdatatest,mechlabelstest,mechlabelstest1)
        print("TestingDNN")
        """# NEED SOMETHING HERE TO TEST THE MODEL ACCURACY

        # also need to clean up the output metrics"""

    elif DNNmodel == "imagebased": # gareths based model
        print("Running DC image based stuff")
        from tensorflow import keras

        imagedimensions = [100,100]
        noise = 0
        n_channels = 1
        dnnBatch = 42
        cpu_number = cpu_workers
        dnnHLNodeNumber = 70
        dnnLearningRate = 1.0e-4
        dnnEpochs = 3
        modeltype = "gareth"
        params = {'dim': (imagedimensions[0], imagedimensions[1]),
                  'batchsize': dnnBatch,
                  'n_classes': Nmodels,  # number of training models
                  'n_channels': n_channels,  # the number of filters present
                  'serverdata': serverdata,
                  'model_numcode': model_numcode,
                  'cpu_number': cpu_number,
                  'exp_data': exp_data,
                  'modeltype': modeltype,
                  "noise": noise}

        #Creat a dic of Estart and Eend

        filename = IB_NNt.genericoutpufile("Imagebased", modeltype)  # input is name of file

        # do an initial thing here to move modeldata to image then save as image
        """GET THE CURRENT TRAIN AND TEST DATA AND get the Estart and Eend values FROM THE SQL"""
        #done above do not uncomment
        #currentdatatest, mechaccepttest = TDNN.sqlcurrentcollector(serverdata, harmnum, harmtest, deci)
        #currentdatatrain, mechaccepttrain = TDNN.sqlcurrentcollector(serverdata, harmnum, harmtrain, deci)
        # above is (current data, [reaction_ID, Reactionmech])

        """GET THE CURRENT TRAIN AND TEST DATA AND get the Estart and Eend values FROM THE SQL Which can then be passed to the below"""

        """Main changes will be in Machine_learning_class for the image collector and the passing the potential around"""
        potdictrain = IB_NNt.potentialgetter(serverdata,mechaccepttrain)
        potdictest = IB_NNt.potentialgetter(serverdata, mechaccepttest)

        # pass all above data to here
        CNNtrain = IB_NNt.Machine_learning_class(currentdatatrain, mechaccepttrain, potdictrain, -1,**params)  # harmnum,batchsize,imagedimensions,serverdata)
        CNNtest = IB_NNt.Machine_learning_class(currentdatatest, mechaccepttest, potdictest,-1, **params)  # harmnum,batchsize,imagedimensions,serverdata)

        model = IB_NNt.garethmodel(imagedimensions, dnnHLNodeNumber, Nmodels)

        #tf.keras.utils.plot_model(model, to_file=filename + "/model_" + name + ".png", show_shapes=True)

        adam_opt = keras.optimizers.Adam(lr=dnnLearningRate)

        model.compile(optimizer=adam_opt, loss='categorical_crossentropy', metrics=['accuracy'])

        #print(model.summary())
        # For simplicity

        history = model.fit(x=CNNtrain, batch_size=dnnBatch, epochs=dnnEpochs, validation_data=CNNtest,
                            validation_batch_size=dnnBatch, max_queue_size=20)

        NNt.accuracyplot(history, harmnum, filename)

        modelname = filename + "/testmodel_" + str(harmnum) + ".model"
        model.save(modelname)
        # train the bloddy thing

        NNt.genericfitingdata(filename, harmnum, (time.time()-time1)/60, model_numcode, model_numcode, harmtrain_mech)

        # clear all neural network data from system and gpu
        del model  # for avoid any trace on aigen
        # tf.reset_default_graph()  # for being sure
        keras.backend.clear_session()

    else:
        print("ERROR: No DNN model has been classified")
        exit()


print("Completion Time (mins): " + str((time.time()-time1)/60))