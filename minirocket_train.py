# Angus Dempster, Daniel F Schmidt, Geoffrey I Webb

# MiniRocket: A Very Fast (Almost) Deterministic Transform for Time Series
# Classification

# https://arxiv.org/abs/2012.08791
import utils
from sklearn.metrics import roc_auc_score
import copy
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from models.minrocket import fit, transform
from dataset import DownLoadECGData
import random
from dataset import hf_dataset


def setup_seed(seed):
    print('seed: ', seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train(num_classes, training_size, X_training, Y_training, X_validation, Y_validation, **kwargs):
    # -- init ------------------------------------------------------------------

    # default hyperparameters are reusable for any dataset
    args = \
        {
            "num_features": 10_000,
            "minibatch_size": 256,
            "lr": 1e-4,
            "max_epochs": 50,
            "patience_lr": 5,  # 50 minibatches
            "patience": 10,  # 100 minibatches
            "cache_size": training_size  # set to 0 to prevent caching
        }
    args = {**args, **kwargs}

    _num_features = 84 * (args["num_features"] // 84)

    def init(layer):
        if isinstance(layer, nn.Linear):
            nn.init.constant_(layer.weight.data, 0)
            nn.init.constant_(layer.bias.data, 0)

    # -- model -----------------------------------------------------------------

    model = nn.Sequential(nn.Linear(_num_features, num_classes))
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args["lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, min_lr=1e-8, patience=args["patience_lr"])
    model.apply(init)

    # -- data -------------------------------------------------------
    X_training, Y_training = X_training.astype(np.float32), torch.FloatTensor(Y_training)
    X_validation, Y_validation = X_validation.astype(np.float32), torch.FloatTensor(Y_validation)
    # -- run -------------------------------------------------------------------

    minibatch_count = 0
    best_validation_loss = np.inf
    stall_count = 0
    stop = False

    print("Training... (faster once caching is finished)")

    for epoch in range(args["max_epochs"]):

        print(f"Epoch {epoch + 1}...".ljust(80, " "), end="\r", flush=True)

        if epoch == 0:
            parameters = fit(X_training, args["num_features"])

            # transform validation data
            X_validation_transform = transform(X_validation, parameters)

        # transform training data
        X_training_transform = transform(X_training, parameters)

        if epoch == 0:
            # per-feature mean and standard deviation
            f_mean = X_training_transform.mean(0)
            f_std = X_training_transform.std(0) + 1e-8

            # normalise validation features
            X_validation_transform = (X_validation_transform - f_mean) / f_std
            X_validation_transform = torch.FloatTensor(X_validation_transform)

        # normalise training features
        X_training_transform = (X_training_transform - f_mean) / f_std
        X_training_transform = torch.FloatTensor(X_training_transform)

        minibatches = torch.randperm(len(X_training_transform)).split(args["minibatch_size"])

        # train on transformed features
        for minibatch_index, minibatch in enumerate(minibatches):

            if epoch > 0 and stop:
                break

            if minibatch_index > 0 and len(minibatch) < args["minibatch_size"]:
                break

            # -- training --------------------------------------------------

            optimizer.zero_grad()
            _Y_training = model(X_training_transform[minibatch])
            training_loss = loss_function(_Y_training, Y_training[minibatch])
            training_loss.backward()
            optimizer.step()

            minibatch_count += 1

            if minibatch_count % 10 == 0:

                _Y_validation = model(X_validation_transform)
                validation_loss = loss_function(_Y_validation, Y_validation)

                scheduler.step(validation_loss)

                if validation_loss.item() >= best_validation_loss:
                    stall_count += 1
                    if stall_count >= args["patience"]:
                        stop = True
                        print(f"\n<Stopped at Epoch {epoch + 1}>")
                else:
                    best_validation_loss = validation_loss.item()
                    best_model = copy.deepcopy(model)
                    if not stop:
                        stall_count = 0

    return parameters, best_model, f_mean, f_std


def predict(parameters, model, f_mean, f_std, X_test, Y_test, **kwargs):
    predictions = []

    X_test = X_test.astype(np.float32)

    X_test_transform = transform(X_test, parameters)
    X_test_transform = (X_test_transform - f_mean) / f_std
    X_test_transform = torch.FloatTensor(X_test_transform)

    _predictions = torch.sigmoid(model(X_test_transform)).cpu().detach().numpy()
    predictions.append(_predictions)
    predictions = np.array(predictions).squeeze(axis=0)
    auc = roc_auc_score(Y_test, predictions)
    TPR = utils.compute_TPR(Y_test, predictions)
    print("AUC = ", auc, "TPR = ", TPR)


def main(data_name='ptbxl'):
    setup_seed(7)
    if data_name == 'ptbxl':
        # eg. ['exp0', 'exp1', 'exp1.1', 'exp1.1.1', 'exp2', 'exp3']
        ded = DownLoadECGData('exp0', 'rhythm', '../data/ptbxl/')
        X_training, Y_training, X_validation, Y_validation, X_test, Y_test = ded.preprocess_data()
    elif data_name == 'cpsc':
        ded = DownLoadECGData('exp_cpsc', 'all', '../data/CPSC/')
        X_training, Y_training, X_validation, Y_validation, X_test, Y_test = ded.preprocess_data()
    else:
        X_training, Y_training, X_validation, Y_validation, X_test, Y_test = hf_dataset()

    parameters, best_model, f_mean, f_std = train(len(Y_training[0]), len(X_training),
                                                  X_training, Y_training,
                                                  X_validation, Y_validation)

    predict(parameters, best_model, f_mean, f_std, X_test, Y_test)

if __name__ == '__main__':
    main()
