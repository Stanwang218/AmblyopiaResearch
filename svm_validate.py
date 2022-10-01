from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from torch.utils.data import DataLoader
from AmblyopiaDatasetPaper import *
import torch
from estcnn import *
import numpy as np

data_length = [250,500, 1000, 1500, 2000, 2500, 3000]
ratio = [0.5, 1, 2, 3, 4, 5, 6]
epoch_list = [10, 20, 30, 50, 500]
# epoch_list = [500]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset_path = r"E:\Dataset\amplyopia"
for length in data_length:
    dataset = AmblyopiaDataset(dataset_path, length)
    batch_size = 16
    data = DataLoader(dataset, batch_size)
    x_list = []
    y_list = []
    for x, y in data:
        x_list.extend(x.numpy().tolist())
        y_list.extend(y.numpy().tolist())
    x = np.array(x_list)
    x = x.reshape(x.shape[0], -1)
    print(x.shape)
    y = np.array(y_list)
    # clf = svm.SVC(probability=True)
    clf = LogisticRegression(max_iter=5000)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.7)
    print(1)
    clf.fit(x_train, y_train)
    print(2)
    pred = clf.predict(x_test)
    print(metrics.accuracy_score(y_test, pred))
    print(metrics.classification_report(y_test, pred))