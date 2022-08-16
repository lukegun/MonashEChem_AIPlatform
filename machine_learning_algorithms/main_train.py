import pandas as pd

import DNNtimecluster_mod as TC_mod
import NN_train_mod
import TrainDNNtimeclass_mod as TDNN
import DC_neuralnetwork_mod as NNt
import ImagebasedDNN as IB_NNt
import time
import sys
import numpy as np
import shutil
import DNN_run_mods as runner_mod
from joblib import dump
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=1424)])


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
                for i in range(1,int(len(values)/2)):
                    s += "_H" + str(i) + values[int(i*2):int(i*2)+2]

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

        """print the react class to file"""
        #print(react_class)
        #print(len(react_class))
        clusteredlist = []
        for stuff in react_class:
            labelnums = []
            for sims in stuff:
                labelnums.append(sims[0])

            bayesdic = TDNN.classifierlabelbayes(labelnums, serverdata,preclusterreactionmech) # last value passes
            clusteredlist.append(bayesdic)

            """something to get the reaction mechanism of the abbove array"""
        #  below has been shown to be consistent

        #print clustered list to each label
        """convert to something we can use for bayesian inference"""


    else: # """something to get the DC clustered label"""
        """THERE A BUG HERE WITH HOW THE SYSTEM REPRESENTS THE NUMBERS IN THE OUTPUT FILE BUT SHOULDN'T EFFECT THE SYSTEM"""
        """BUG IS PURELY VISUAL AND IN THE OUTPUT BAYES PRINTER FIX NOT REQUIRED FOR DC AS ALL UNIVARIANT DATA"""
        preclusterreactionmech = reactionmech
        preclusterreact_class = react_class
        react_class, reactionmech = TDNN.DCreactclusterupdate(react_class,serverdata)
        #print(react_class)

        """print the react class to file"""
        holdtot = []
        nnn = len(reactionmech)
        [holdtot.append([]) for i in range(nnn)]
        for stuff in react_class:
            for x in stuff: # itterate through [25016, 'H-1ad'],
                for j in range(nnn):
                    # sort into the system based on the cluster
                    if x[1] == reactionmech[j]:
                        holdtot[j].append(x[0])
                        break
        labelnums = holdtot


        # print(react_class)
        # print(len(react_class))
        clusteredlist = []
        for i in range(nnn):
            print(len(labelnums[i]))
            bayesdic = TDNN.classifierlabelbayes(labelnums[i], serverdata, preclusterreactionmech)  # last value passes
            clusteredlist.append(bayesdic)
        print(clusteredlist)


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

Nmodels = len(reactionmech)


#SAVE THE MODEL to file with settings
filename = supivisorclass+"_"+DNNmodel +"_Var"+str(variate)
#make file
filename = NN_train_mod.genericoutpufile(supivisorclass+"_"+DNNmodel +"_Var"+str(variate))
#copy settings
modelname = filename+ "/trained.model"
shutil.copyfile(input_name, filename+"/settings.txt")

if supivisorclass == "clustering":
    #  below has been shown to be consistent
    f = open(filename+"/clusteringlabelsbayes.txt","w+")
    for keys, items in model_numcode.items():
        f.write(str(items)+"\t")
        for key,item in clusteredlist[items].items():
            f.write(key+":"+str(item)+"\t")
        f.write("\n")
    f.close()

    #creates a file for sublabels
    if ACCase:
        f = open(filename+"/clusteringnumberlabelinfo.txt","w+")

        for keys,items in model_numcode.items():
            f.write("Label# "+str(items)+"\t"+"numsims:"+str(len(react_class[items]))+"\t"+keys+": ")
            # gets out all the seperate labels
            x  = groupeddic.get(keys)
            for stuff in x:
                f.write(stuff + "\t")
            f.write("\n")
        f.close()

    #print clustered list to each label
    """convert to something we can use for bayesian inference"""

"""PUT ANALYSIS PLOTS IN THE ABOVE FILE"""

#seperates the input data into each model type
if variate: #use the multivariate models TRUE = multivariate
    #traindata = traindata[:200]
    #testdata = testdata[:40]

    print("Running classifier on AC data")
    harmtrain, harmtrain_mech, harmtrain_ID = NNt.AC_NN_setter(traindata, harmdata)  # training data
    harmtest, harmtest_mech, harmtest_ID = NNt.AC_NN_setter(testdata, harmdata)  # testing data for validation

    #harmtest = harmtest[0:90]
    #harmtrain = harmtrain[0:900]

    # small change made fater less likely to parralellise it
    currentdatatest, mechaccepttest = TDNN.ACsqlcurrentcollector(serverdata, harmdata, harmtest, deci,DNNmodel)
    currentdatatrain, mechaccepttrain = TDNN.ACsqlcurrentcollector(serverdata, harmdata, harmtrain, deci,DNNmodel)

    #This is to do the corrections on surface concentration and to make the normalisation the same
    if True:
        scalar = 7568546.0  # average scalar difference between E reaction mechanism and Esurf
        if supivisorclass == "reactmech":
            for i in range(len(mechaccepttest)):
                if mechaccepttest[i][1] == "ESurf":
                    currentdatatest[i] = currentdatatest[i]*scalar

            for i in range(len(mechaccepttrain)):
                if mechaccepttrain[i][1] == "ESurf":
                    currentdatatrain[i] = currentdatatrain[i]*scalar
        elif supivisorclass == "clustering": # this is required due to the clustering
            # change in 1 and 2 is a quick fix for the translations
            for i in range(len(mechaccepttest)):
                if mechaccepttest[i][2] == "ESurf":
                    currentdatatest[i] = currentdatatest[i] * scalar

            for i in range(len(mechaccepttrain)):
                if mechaccepttrain[i][2] == "ESurf":
                    currentdatatrain[i] = currentdatatrain[i] * scalar

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
        print(currentdatatrain.shape)
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

        # use a voltage scale
        import matplotlib.pyplot as plt

        testVscale = False
        if testVscale:
            Nchanel = 2

            deci = int(deci / 2)
            plt.plot(currentdatatrain[0, :])
            currentdatatrain = np.array([currentdatatrain[:, :deci], currentdatatrain[:, -1:deci - 1:-1]])
            currentdatatest = np.array([currentdatatest[:, :deci], currentdatatest[:, -1:deci - 1:-1]])
            plt.plot(currentdatatrain[0, 0, :])
            plt.plot(currentdatatrain[1, 0, :])
            nchanels, ntrain, r = currentdatatrain.shape
            nchanels, ntest, r = currentdatatest.shape
            print(currentdatatrain.shape)

            currentdatatrain = currentdatatrain.transpose(1, 2, 0)
            currentdatatest = currentdatatest.transpose(1, 2, 0)

        else:
            pass
            """ntrain, r,Nchanel = currentdatatrain.shape
            ntest, r,Nchanel = currentdatatest.shape

            currentdatatrain = currentdatatrain.reshape(ntrain, r, Nchanel)
            currentdatatest = currentdatatest.reshape(ntest, r, Nchanel)"""
        print(currentdatatrain.shape)


        # these are 1D so not required
        mechlabelstrain = np.array(mechlabelstrain)

        # sets up the class weights for the uneven groups
        class_weight = {}
        u, i = np.unique(mechlabelstrain, return_counts=True)
        dictnum = dict(zip(u, i))

        f = max(dictnum, key=dictnum.get)
        f = dictnum[f]
        for keys, items in dictnum.items():
            print(items)
            if supivisorclass == "clustering":
                class_weight[keys] = f / items
                lr = 0.1 #0.5
                Nepoch = 40
                adjustable_lr0 = True

            else:
                class_weight[keys] = 1
                lr = 0.5
                Nepoch = 40
                adjustable_lr0 = True
        print(class_weight)

        print("Data Collected fitting DNN")

        # Needs to be after the weights
        mechlabelstrain = keras.utils.to_categorical(mechlabelstrain, num_classes=Nmodels)
        mechlabelstest1 = np.array(mechlabelstest)
        mechlabelstest = keras.utils.to_categorical(mechlabelstest, num_classes=Nmodels)


        # model = Classifier_INCEPTION("output",None,nb_classes=Nmodels )

        model = Classifier_INCEPTION(filename + "/outputAC", (deci, Nchanel), nb_classes=Nmodels, verbose=True,lr0 = lr,
                                     nb_epochs=Nepoch,depth=4,adjustable_lr0=adjustable_lr0,mode="AC")  # using 20 epochs for ease depth=6 lr0= 0.5, nb_epochs=40,depth=4
        model.build_model((deci, Nchanel), Nmodels)
        model.fit(currentdatatrain, mechlabelstrain, currentdatatest, mechlabelstest, mechlabelstest1,plot_test_acc=True, class_weight=class_weight)

        """# NEED SOMETHING HERE TO TEST THE MODEL ACCURACY

            # also need to clean up the output metrics"""

    else:
        print("ERROR: No DNN model has been classified")
        exit()

else: # use the univarate models Mainly used in DC Case
    #traindata = traindata[:200]
    #testdata = testdata[:40]
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

        rocket = Rocket(num_kernels=5000)  # by default, ROCKET uses 10,000 kernels ideally but where trying to get ram working
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

        # use a voltage scale
        import matplotlib.pyplot as plt
        testVscale = False
        if testVscale:
            Nchanel = 2

            deci = int(deci/2)
            plt.plot(currentdatatrain[0, :])
            currentdatatrain = np.array([currentdatatrain[:, :deci], currentdatatrain[:, -1:deci - 1:-1]])
            currentdatatest = np.array([currentdatatest[:, :deci], currentdatatest[:, -1:deci-1:-1]])
            plt.plot(currentdatatrain[0, 0,:])
            plt.plot(currentdatatrain[1, 0, :])
            nchanels,ntrain, r = currentdatatrain.shape
            nchanels,ntest, r = currentdatatest.shape
            print(currentdatatrain.shape)

            currentdatatrain = currentdatatrain.transpose(1, 2, 0)
            currentdatatest = currentdatatest.transpose(1, 2, 0)

        else:
            Nchanel = 1
            ntrain,r = currentdatatrain.shape
            ntest, r = currentdatatest.shape

            currentdatatrain = currentdatatrain.reshape(ntrain,r,Nchanel)
            currentdatatest = currentdatatest.reshape(ntest, r, Nchanel)

        print("double check the univariate DNN case")
        """print(currentdatatrain.shape)
        plt.plot(currentdatatrain[0,:,0])
        plt.plot(currentdatatrain[0, :, 1])
        plt.savefig("test.png")
        plt.close()
        exit()"""

        # these are 1D so not required
        mechlabelstrain = np.array(mechlabelstrain)

        # sets up the class weights for the uneven groups
        class_weight = {}
        u, i = np.unique(mechlabelstrain, return_counts=True)
        dictnum = dict(zip(u, i))

        f = max(dictnum, key=dictnum.get)
        f = dictnum[f]
        for keys, items in dictnum.items():
            print(items)
            if supivisorclass == "clustering":
                class_weight[keys] = 2*f / items
                lr = 0.1
                Nepoch = 30
                adjustable_lr0 = True
            else:
                class_weight[keys] = 1
                lr = 0.05
                Nepoch = 20
                adjustable_lr0 = False
        print(class_weight)

        mechlabelstrain = keras.utils.to_categorical(mechlabelstrain, num_classes=Nmodels)
        mechlabelstest1 = np.array(mechlabelstest)
        mechlabelstest = keras.utils.to_categorical(mechlabelstest1, num_classes=Nmodels)

        print("Data Collected fitting DNN")

        #model = Classifier_INCEPTION("output",None,nb_classes=Nmodels )

        model = Classifier_INCEPTION(filename + "/outputDC", (deci,Nchanel), nb_classes=Nmodels,verbose=True,nb_epochs=Nepoch,lr0 = lr, # 0.5 for clustering 0.1 for react mech
                                     depth=3,nb_filters=64,adjustable_lr0=adjustable_lr0) # using 20 epochs for ease #attempting 4 3 is known to work better ,adjustable_lr0=True
        model.build_model((deci,Nchanel),Nmodels)
        model.fit(currentdatatrain, mechlabelstrain,currentdatatest,mechlabelstest,mechlabelstest1,plot_test_acc=True,class_weight= class_weight)
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

        # train the bloddy thing

        NNt.genericfitingdata(filename, harmnum, (time.time()-time1)/60, model_numcode, model_numcode, harmtrain_mech)

        # clear all neural network data from system and gpu
        del model  # for avoid any trace on aigen
        # tf.reset_default_graph()  # for being sure
        keras.backend.clear_session()

    else:
        print("ERROR: No DNN model has been classified")
        exit()

# Also need something here to do the important values and save to file EG run time #mechanisms #number of stuff # any other shit


# saves the model to file with all information

"""MODELS WILL NEED TO BE SAVED DIFFERENTLY FOR EACH SYSTEM"""
if DNNmodel == "imagebased": # keras based models
    model.save(modelname)

elif  DNNmodel == "inceptiontime":
    """FIX"""
    #saved as part of training model inside the inceptiontime

elif DNNmodel == "rocket": #sktime based models similaur to scikit learn
    from joblib import dump

    dump(rocket, filename+ "/rocket.model")  # might need a correction here for .joblib file
    dump(classifier, modelname) #might need a correction here for .joblib file

print("Completion Time (mins): " + str((time.time()-time1)/60))
exit()
