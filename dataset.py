import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from data_process import load_dataset, compute_label_aggregations, select_data, preprocess_signals, data_slice, hf_dataset
from config import config


class ECGDataset(Dataset):
    """
    A generic data loader where the samples are arranged in this way:
    """

    def __init__(self, signals: np.ndarray, labels: np.ndarray):
        super(ECGDataset, self).__init__()
        self.data = signals
        self.label = labels
        self.num_classes = self.label.shape[1]

        self.cls_num_list = np.sum(self.label, axis=0)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.label[index]

        x = x.transpose()

        x = torch.tensor(x.copy(), dtype=torch.float)

        y = torch.tensor(y, dtype=torch.float)
        y = y.squeeze()
        return x, y

    def __len__(self):
        return len(self.data)


class DownLoadECGData:
    '''
        All experiments data
    '''

    def __init__(self, experiment_name, task, datafolder, sampling_frequency=100, min_samples=0,
                 train_fold=8, val_fold=9, test_fold=10):
        self.min_samples = min_samples
        self.task = task
        self.train_fold = train_fold
        self.val_fold = val_fold
        self.test_fold = test_fold
        self.experiment_name = experiment_name
        self.datafolder = datafolder
        self.sampling_frequency = sampling_frequency

    def preprocess_data(self):
        # Load PTB-XL data
        data, raw_labels = load_dataset(self.datafolder, self.sampling_frequency)
        # Preprocess label data
        labels = compute_label_aggregations(raw_labels, self.datafolder, self.task)

        # Select relevant data and convert to one-hot
        data, labels, Y, _ = select_data(data, labels, self.task, self.min_samples)

        if self.datafolder == '../data/CPSC/':
            data = data_slice(data)

        # 10th fold for testing (9th for now)
        X_test = data[labels.strat_fold == self.test_fold]
        y_test = Y[labels.strat_fold == self.test_fold]
        # 9th fold for validation (8th for now)
        X_val = data[labels.strat_fold == self.val_fold]
        y_val = Y[labels.strat_fold == self.val_fold]
        # rest for training
        X_train = data[labels.strat_fold <= self.train_fold]
        y_train = Y[labels.strat_fold <= self.train_fold]

        # Preprocess signal data
        X_train, X_val, X_test = preprocess_signals(X_train, X_val, X_test)

        return X_train, y_train, X_val, y_val, X_test, y_test


def load_datasets(datafolder=None, experiment=None):
    '''
    Load the final dataset
    '''
    experiment = experiment

    if datafolder == '../data/ptbxl/':
        experiments = {
            'exp0': ('exp0', 'all'),
            'exp1': ('exp1', 'diagnostic'),
            'exp1.1': ('exp1.1', 'subdiagnostic'),
            'exp1.1.1': ('exp1.1.1', 'superdiagnostic'),
            'exp2': ('exp2', 'form'),
            'exp3': ('exp3', 'rhythm')
        }
        name, task = experiments[experiment]
        ded = DownLoadECGData(name, task, datafolder)
        X_train, y_train, X_val, y_val, X_test, y_test = ded.preprocess_data()
    elif datafolder == '../data/CPSC/':
        ded = DownLoadECGData('exp_CPSC', 'all', datafolder)
        X_train, y_train, X_val, y_val, X_test, y_test = ded.preprocess_data()
    else:
        X_train, y_train, X_val, y_val, X_test, y_test = hf_dataset(datafolder)

    ds_train = ECGDataset(X_train, y_train)
    ds_val = ECGDataset(X_val, y_val)
    ds_test = ECGDataset(X_test, y_test)

    num_classes = ds_train.num_classes
    train_dataloader = DataLoader(ds_train, batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(ds_val, batch_size=config.batch_size, shuffle=False)
    test_dataloader = DataLoader(ds_test, batch_size=config.batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader, num_classes


