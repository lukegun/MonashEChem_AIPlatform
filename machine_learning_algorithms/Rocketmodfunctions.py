
import torch, torch.nn as nn, torch.optim as optim # MAY NEED TO DO THIS IF R DOES NOT SCALE
import numpy as np
from rocket_functions import apply_kernels, generate_kernels

# == training function =========================================================
# this is for rocket scalability and ram usage of the ridge classifier
def train(X,
          Y,
          X_validation,
          Y_validation,
          kernels,
          num_features,
          num_classes,
          minibatch_size = 256,
          max_epochs = 100,
          patience = 2,           # x10 minibatches; reset if loss improves
          tranche_size = 2 ** 11,
          cache_size = 2 ** 14):  # as much as possible

    # -- init ------------------------------------------------------------------

    def init(layer):
        if isinstance(layer, nn.Linear):
            nn.init.constant_(layer.weight.data, 0)
            nn.init.constant_(layer.bias.data, 0)

    # -- model -----------------------------------------------------------------

    model = nn.Sequential(nn.Linear(num_features, num_classes)) # logistic / softmax regression
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.5, min_lr = 1e-8)
    model.apply(init)

    # -- run -------------------------------------------------------------------

    minibatch_count = 0
    best_validation_loss = np.inf
    stall_count = 0
    stop = False

    num_examples = len(X)
    num_tranches = np.int(np.ceil(num_examples / tranche_size))

    cache = np.zeros((min(cache_size, num_examples), num_features))
    cache_count = 0

    for epoch in range(max_epochs):

        if epoch > 0 and stop:
            break

        for tranche_index in range(num_tranches):

            if epoch > 0 and stop:
                break

            a = tranche_size * tranche_index
            b = a + tranche_size

            Y_tranche = Y[a:b]

            # if cached, use cached transform; else transform and cache the result
            if b <= cache_count:

                X_tranche_transform = cache[a:b]

            else:

                X_tranche = X[a:b]
                #X_tranche = (X_tranche - X_tranche.mean(axis = 1)) / X_tranche.std(axis = 1) # normalise time series
                X_tranche_transform = apply_kernels(X_tranche, kernels)

                if epoch == 0 and tranche_index == 0:

                    # per-feature mean and standard deviation (estimated on first tranche)
                    f_mean = X_tranche_transform.mean(0)
                    f_std = X_tranche_transform.std(0) + 1e-8

                    # normalise and transform validation data
                    #X_validation = (X_validation - X_validation.mean(axis = 1, keepdims = True)) / X_validation.std(axis = 1, keepdims = True) # normalise time series
                    X_validation_transform = apply_kernels(X_validation, kernels)
                    X_validation_transform = (X_validation_transform - f_mean) / f_std # normalise transformed features
                    X_validation_transform = torch.FloatTensor(X_validation_transform)
                    Y_validation = torch.LongTensor(Y_validation)

                X_tranche_transform = (X_tranche_transform - f_mean) / f_std # normalise transformed features

                if b <= cache_size:

                    cache[a:b] = X_tranche_transform
                    cache_count = b

            X_tranche_transform = torch.FloatTensor(X_tranche_transform)
            Y_tranche = torch.LongTensor(Y_tranche)

            minibatches = torch.randperm(len(X_tranche_transform)).split(minibatch_size)

            for minibatch_index, minibatch in enumerate(minibatches):

                if epoch > 0 and stop:
                    break

                # abandon undersized minibatches
                if minibatch_index > 0 and len(minibatch) < minibatch_size:
                    break

                # -- (optional) minimal lr search ------------------------------

                # default lr for Adam may cause training loss to diverge for a
                # large number of kernels; lr minimising training loss on first
                # update should ensure training loss converges

                if epoch == 0 and tranche_index == 0 and minibatch_index == 0:

                    candidate_lr = 10 ** np.linspace(-1, -6, 6)

                    best_lr = None
                    best_training_loss = np.inf

                    for lr in candidate_lr:

                        lr_model = nn.Sequential(nn.Linear(num_features, num_classes))
                        lr_optimizer = optim.Adam(lr_model.parameters())
                        lr_model.apply(init)

                        for param_group in lr_optimizer.param_groups:
                            param_group["lr"] = lr

                        # perform a single update
                        lr_optimizer.zero_grad()
                        Y_tranche_predictions = lr_model(X_tranche_transform[minibatch])
                        training_loss = loss_function(Y_tranche_predictions, Y_tranche[minibatch])
                        training_loss.backward()
                        lr_optimizer.step()

                        Y_tranche_predictions = lr_model(X_tranche_transform)
                        training_loss = loss_function(Y_tranche_predictions, Y_tranche).item()

                        if training_loss < best_training_loss:
                            best_training_loss = training_loss
                            best_lr = lr

                    for param_group in optimizer.param_groups:
                        param_group["lr"] = best_lr

                # -- training --------------------------------------------------

                optimizer.zero_grad()
                Y_tranche_predictions = model(X_tranche_transform[minibatch])
                training_loss = loss_function(Y_tranche_predictions, Y_tranche[minibatch])
                training_loss.backward()
                optimizer.step()

                minibatch_count += 1

                if minibatch_count % 10 == 0:

                    Y_validation_predictions = model(X_validation_transform)
                    validation_loss = loss_function(Y_validation_predictions, Y_validation)

                    scheduler.step(validation_loss)

                    if validation_loss.item() >= best_validation_loss:
                        stall_count += 1
                        if stall_count >= patience:
                            stop = True
                    else:
                        best_validation_loss = validation_loss.item()
                        if not stop:
                            stall_count = 0

    return model, f_mean, f_std
