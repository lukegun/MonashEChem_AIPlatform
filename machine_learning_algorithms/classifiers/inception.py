# resnet model
from tensorflow import keras
import numpy as np
import time

from utils.utils import save_logs
from utils.utils import calculate_metrics
from utils.utils import save_test_duration


class Classifier_INCEPTION:

    def __init__(self, output_directory, input_shape, nb_classes, verbose=False, build=True, adjustable_lr0=False,batch_size=64,lr0=0.01,
                 nb_filters=64, use_residual=True, use_bottleneck=True, depth=6, kernel_size=41, nb_epochs=1500,denseact = "softmax",mode="DC"):

        self.output_directory = output_directory

        self.nb_filters = nb_filters
        self.use_residual = use_residual
        self.use_bottleneck = use_bottleneck
        self.depth = depth
        self.kernel_size = kernel_size - 1
        self.callbacks = None
        self.batch_size = batch_size
        self.bottleneck_size = 32
        self.nb_epochs = nb_epochs
        self.lr0 = lr0
        self.adjustable_lr0 = adjustable_lr0
        self.denseact = denseact
        self.mode = mode

        if build == True:
            if mode == "DC":
                self.model = self.build_model(input_shape, nb_classes)
            else:
                self.model = self.build_model(input_shape, nb_classes,metricuse="accuracy")
            if (verbose == True):
                self.model.summary()
            self.verbose = verbose
            self.model.save_weights(self.output_directory + 'model_init.hdf5')

    def _inception_module(self, input_tensor, stride=1, activation='linear'):

        if self.use_bottleneck and int(input_tensor.shape[-1]) > 1:
            input_inception = keras.layers.Conv1D(filters=self.bottleneck_size, kernel_size=1,
                                                  padding='same', activation=activation, use_bias=False)(input_tensor)
        else:
            input_inception = input_tensor
        #kernel_size_s = [3, 5, 8, 11, 17,28,40] #WAS USED FOR REACTION MECHANISMS
        if self.mode == "DC":
            kernel_size_s = [3, 5, 9, 11, 17, 27, 41]
        else:
            #kernel_size_s = [3, 5, 9, 11, 17, 27, 41]
            kernel_size_s = [self.kernel_size // (2 ** i) + 1 for i in range(3)] # THIS IS USED FOR CLUSTERING

        # kernel_size_s = [self.kernel_size // (2 ** i) for i in range(3)]

        conv_list = []
        for i in range(len(kernel_size_s)):
            conv_list.append(keras.layers.Conv1D(filters=self.nb_filters, kernel_size=kernel_size_s[i],
                                                 strides=stride, padding='same', activation=activation, use_bias=False)(
                input_inception))

        #pool_size = 3 originally

        max_pool_1 = keras.layers.MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)

        conv_6 = keras.layers.Conv1D(filters=self.nb_filters, kernel_size=1,
                                     padding='same', activation=activation, use_bias=False)(max_pool_1)

        conv_list.append(conv_6)

        x = keras.layers.Concatenate(axis=2)(conv_list)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(activation='relu')(x)
        return x

    def _shortcut_layer(self, input_tensor, out_tensor):
        shortcut_y = keras.layers.Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1,
                                         padding='same', use_bias=False)(input_tensor)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        x = keras.layers.Add()([shortcut_y, out_tensor])
        x = keras.layers.Activation('relu')(x)
        return x

    def build_model(self, input_shape, nb_classes,metricuse='accuracy'):
        input_layer = keras.layers.Input(input_shape)

        x = input_layer
        input_res = input_layer

        for d in range(self.depth):

            x = self._inception_module(x)

            if self.use_residual and d % 3 == 2:
                x = self._shortcut_layer(input_res, x)
                input_res = x

        gap_layer = keras.layers.GlobalAveragePooling1D()(x)

        #FIX BELOW TO
        output_layer = keras.layers.Dense(nb_classes, activation=self.denseact)(gap_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=self.lr0),
                      metrics=[metricuse]) # OG learning_rate=0.01 ,'val_accuracy'

        #reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50,
        #                                              min_lr=0.0001)

        file_path = self.output_directory + 'best_model.hdf5'

        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='val_loss',
                                                           save_best_only=True)

        if self.adjustable_lr0:
            if self.mode == "AC":
                loss_tracer  = loss_rollback(model, False, 1.0, True)
                reduce_lr = keras.callbacks.LearningRateScheduler(self.scheduler)
                self.callbacks = [model_checkpoint, reduce_lr]
            else:
                reduce_lr = keras.callbacks.LearningRateScheduler(self.schedulerDC)
                self.callbacks = [model_checkpoint,reduce_lr]  # [reduce_lr, model_checkpoint]
        else:
            self.callbacks = [model_checkpoint]

        return model

    def scheduler(self, epoch, lr):

        if epoch == 0 or epoch == 1 :
            lr = lr
        elif epoch == 5: # try 20
            lr = lr * 0.33**1
        elif epoch%10 == 0 and epoch <40: # try 20
            lr = lr * 0.33**1#np.floor(epoch /5) #0.33
        else:
            lr = lr
        return lr

    def schedulerDC(self, epoch, lr):

        if epoch< 10 :
            lr = self.lr0
        else: # try 20
            lr = self.lr0 * 0.33**np.floor((epoch-5)/5) #0.33

        return lr

    def fit(self, x_train, y_train, x_val, y_val, y_true, plot_test_acc=False,class_weight = {}):
	# Removed check as we are always assuming GPU LG
        #if len(keras.backend.tensorflow_backend._get_available_gpus()) == 0:
        #    print('error no gpu')
        #    exit()
        # x_val and y_val are only used to monitor the test loss and NOT for training


        if self.batch_size is None:
            mini_batch_size = int(min(x_train.shape[0] / 10, 16))
        else:
            mini_batch_size = self.batch_size

        start_time = time.time()

        if plot_test_acc:

            hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=self.nb_epochs,
                                  verbose=self.verbose, validation_data=(x_val, y_val), callbacks=self.callbacks, class_weight=class_weight)
        else:

            hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=self.nb_epochs,
                                  verbose=self.verbose, callbacks=self.callbacks, class_weight=class_weight)

        duration = time.time() - start_time

        self.model.save(self.output_directory + 'last_model.hdf5')

        y_pred = self.predict(x_val, y_true, x_train, y_train, y_val,
                              return_df_metrics=False)

        # save predictions
        np.save(self.output_directory + 'y_pred.npy', y_pred)

        # convert the predicted from binary to integer
        y_pred = np.argmax(y_pred, axis=1)

        df_metrics = save_logs(self.output_directory, hist, y_pred, y_true, duration,
                               plot_test_acc=plot_test_acc)

        keras.backend.clear_session()

        return df_metrics

    def predict(self, x_test, y_true, x_train, y_train, y_test, return_df_metrics=True):
        start_time = time.time()
        model_path = self.output_directory + 'best_model.hdf5'
        model = keras.models.load_model(model_path)
        y_pred = model.predict(x_test, batch_size=self.batch_size)
        if return_df_metrics:
            y_pred = np.argmax(y_pred, axis=1)
            df_metrics = calculate_metrics(y_true, y_pred, 0.0)
            return df_metrics
        else:
            test_duration = time.time() - start_time
            save_test_duration(self.output_directory + 'test_duration.csv', test_duration)
            return y_pred

# this is to fix the stability issue with the AC methods
class loss_rollback(keras.callbacks.Callback):
    def __init__(self,model, monitor_acc,  factor, verbose):
        super(loss_rollback, self).__init__()
        self.model=model
        self.initial_lr=float(keras.backend.get_value(model.optimizer.lr)) # get the initiallearning rate and save it
        self.lowest_vloss=np.inf # set lowest validation loss to infinity initially
        self.best_weights=self.model.get_weights() # set best weights to model's initial weights
        self.verbose=verbose
        self.monitor_acc= monitor_acc
        self.highest_acc=0
        self.factor = factor
    def on_epoch_end(self, epoch, logs=None):  # method runs on the end of each epoch
        lr=float(keras.backend.get_value(self.model.optimizer.lr)) # get the current learning rate
        vloss=logs.get('val_loss')  # get the validation loss for this epoch

        acc=logs.get('accuracy')
        if self.monitor_acc==False: # monitor validation loss
            if vloss>self.lowest_vloss*1.5 and epoch > 15:
                self.model.set_weights(self.best_weights)
                new_lr=lr * self.factor
                keras.backend.set_value(self.model.optimizer.lr, new_lr)
                if self.verbose:
                    print( '\n model weights reset to best weights and reduced lr to ', new_lr)
            elif vloss<self.lowest_vloss*1.1:
                self.best_weights = self.model.get_weights()
                self.lowest_vloss = vloss
                if self.verbose:
                    print( '\n New model set as the best fit , value ', vloss)

