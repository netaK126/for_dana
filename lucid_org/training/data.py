import pandas as pd
import torch
import math

# includes functions for preprocessing the data, and a data loader.

class DataLoader():
    def __init__(self, dataset, batch_size=1, shuffle=False):
        super(DataLoader, self).__init__()
        data, labels = dataset
        self.dataset = dataset
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.num_samples = len(labels)
        self.num_batches = math.ceil(self.num_samples / batch_size)
        self.extra = self.num_batches * batch_size - self.num_samples
        self.shuffle = shuffle
        self.idx = self.reset()
        self.batch = 0
    def __len__(self):
        return len(self.data)

    def reset(self):
        if self.shuffle:
            idx = torch.randperm(self.num_samples)
        else:
            idx = torch.arange(self.num_samples)
        if self.extra > 0:
            idx = torch.cat((idx, -1 * torch.ones(self.extra).long()))
        return idx.view(self.num_batches, self.batch_size)

    def get_batch(self):
        idx = self.idx[self.batch]
        idx = idx[idx >= 0]
        data = self.data[idx]
        labels = self.labels[idx]
        self.batch += 1
        if self.batch == self.num_batches:
            self.idx = self.reset()
            self.batch = 0
            last = True
        else:
            last = False
        return data, labels, last

def preprocess(path, dataset):
    """
    preprocess the given dataset.
    """
    if dataset == 'adult':
        df, names, labels = get_adult(path)
    elif dataset == 'bank':
        df, names, labels = get_bank(path)
    elif dataset == 'credit':
        df, names, labels = get_credit(path)
    else:
        print('Invalid dataset')
        exit()
    data = ()
    for i in range(len(names)):
        data = data + (encode(df[names[i]], labels[i]),)
        print(data[-1].min().item(), data[-1].max().item())
        print(names[i])
    data = torch.cat(data, dim=1)
    return data[:, :-1], 1 - 2 * data[:, -1].int()

def get_adult(path):
    """
    preprocess the adult dataset.
    """
    names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'martial-status', 'occupation', 'relationship',
             'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
    df = pd.read_csv(path, sep=", ", header=None, comment='.', names=names)
    labels = [
        [100.],
        ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov',
         'Without-pay', 'Never-worked'],
        [1500000.],
        ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th',
         '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'],
        [16.],
        ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed',
         'Married-spouse-absent', 'Married-AF-spouse'],
        ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty',
         'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing',
         'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'],
        ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'],
        ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'],
        ['Female', 'Male'],
        [100000.], [5000.], [100.],
        ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)',
         'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland',
         'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador',
         'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand',
         'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands'],
        ['>50K', '<=50K']
    ]
    return df, names, labels

def get_bank(path):
    """
    preprocess the bank dataset.
    """
    names = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact',
             'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']
    df = pd.read_csv(path, sep=";")
    labels = [
        [100.],
        ['admin.', 'unknown', 'unemployed', 'management', 'housemaid', 'entrepreneur',
         'student', 'blue-collar', 'self-employed', 'retired', 'technician', 'services'],
        ['married', 'divorced', 'single'],
        ['unknown', 'secondary', 'primary', 'tertiary'],
        ['yes', 'no'],
        [120000., 10000.],
        ['yes', 'no'],
        ['yes', 'no'],
        ['unknown', 'telephone', 'cellular'],
        [30., -1.],
        ['jan', 'feb' 'mar', 'apr', 'may'', jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
        [5000.], [64.], [900., 1.], [280.],
        ['unknown', 'other', 'failure', 'success'],
        ['yes', 'no']
    ]
    return df, names, labels

def get_credit(path):
    """
    preprocess the credit dataset.
    """
    names = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
             'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
             'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6', 'DEFAULT']
    df = pd.read_csv(path)
    labels = [
        [1000000.], [1., -1.], [6.], [3.], [80.], [10., 2.], [10., 2.], [10., 2.], [10., 2.], [10., 2.], [10., 2.],
        [2100000., 350000.], [2100000., 350000.], [2100000., 350000.], [2100000., 350000.], [2100000., 350000.],
        [2100000., 350000.], [1800000.], [1800000.], [1800000.], [1800000.], [1800000.], [1800000.], [-1., -1.]
    ]
    return df, names, labels

def encode(df, labels):
    """
    preprocess the given data
    """
    L = len(labels)
    if type(labels[0]) == str:
        if L == 2:
            return torch.tensor(df == labels[1]).float().unsqueeze(1)
        t = torch.zeros(len(df), 2)
        for i in range(L):
            theta = 0.5 * math.pi * i / (L - 1)
            t[df == labels[i], 0] = math.cos(theta)
            t[df == labels[i], 1] = math.sin(theta)
        return t
    t = torch.tensor(df).float()
    print(t.min().item(), t.max().item())
    if L > 1:
        t = t + labels[1]
    return (t / labels[0]).unsqueeze(1)

if __name__ == '__main__':
    # loads, preprocess and save the dataset.
    path = '../datasets/credit/'
    train = 'credit_card.csv'
    data, labels = preprocess(path + train, 'credit')
    N = len(labels)
    n = int(0.3 * N)
    torch.manual_seed(0)
    p = torch.randperm(N)
    train_data = data[p[n:]]
    test_data = data[p[:n]]
    train_labels = labels[p[n:]]
    test_labels = labels[p[:n]]
    print(train_data, train_data.size())
    print(train_labels, train_labels.size())
    print(train_data.mean(dim=0))
    torch.save((train_data, train_labels), path + 'train.pth')
    print(test_data, test_data.size())
    print(test_labels, test_labels.size())
    print(test_data.mean(dim=0))
    torch.save((test_data, test_labels), path + 'test.pth')

