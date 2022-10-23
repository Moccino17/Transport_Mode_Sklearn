import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.special.cython_special import y0
from copy import deepcopy
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import seaborn as sn

pd.options.mode.chained_assignment = None  # default='warn'


def normalize_cols(x):
    for col in x.columns:
        x[col] = (x[col] - x[col].min()) / (x[col].max() - x[col].min())
    return x


def load_transport_dataset():
    data = pd.read_csv('dataset.csv')
    # data = data.loc[data['HH Car Ownership'] == 2]
    data['Passenger'] = data['Passenger'].str.strip()
    data['Service quality'] = data['Service quality'].str.strip()

    # train = data.loc[:, ~data.columns.isin(['Mode', 'Parking Cost', 'Interview no.', 'Service quality', "Passenger"])]
    train = data.loc[:, data.columns.isin(
        ['HH Car Ownership', 'In-Vehicle Trip Time', 'Out-of-Vehicle Trip Time', 'TotalTripCost'])]
    # train = data.loc[:, data.columns.isin(['In-Vehicle Trip Time', 'Out-of-Vehicle Trip Time', 'TotalTripCost'])]
    labels = data['Mode']

    # train['Passenger'] = LabelEncoder().fit_transform(train['Passenger'])
    # train['Service quality'] = LabelEncoder().fit_transform(train['Service quality'])
    return train, labels, data


def print_confusion_matrix(model, x, y):
    print(model.__class__.__name__, ': -------------')
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    print('accuracy = ' + str(1 - (pred - y_test).mean()))
    print(confusion_matrix(y_test, pred))


def print_feature_importance_results(x, y, model):
    print(model.__class__.__name__, ': -------------')
    model.fit(x, y)
    for v in model.feature_importances_:
        print(round(v, 4))
    print('------------------------------------------')

def plot_pca_results(x, y):
    pcs = PCA(n_components=3).fit_transform(x)
    plt.scatter(pcs[:, 0], pcs[:, 1], c=y)
    plt.show()


def plot_correlation_matrix(data):
    corrMatrix = data.corr()
    sn.heatmap(corrMatrix, annot=True)
    plt.show()


def generate_confusion_matrices(models):
    xtrain, xtest, ytrain, ytest = load_train_test_dataset()

    for model in models:
        model.fit(xtrain, ytrain)
        pred = model.predict(xtest)
        print('accuracy = ', accuracy_score(ytest, pred))
        print(confusion_matrix(ytest, pred))


def sensitivity_accuracy_vs_data_size(models):
    x1, y1, _ = load_transport_dataset()
    for model in models:
        print(model.__class__.__name__)
        for size in range(1, 100):
            x, xt, y, yt = train_test_split(x1, y1, train_size = size/100)
            print(round(cross_val_score(deepcopy(model), x, y).mean(), 5))


classifiers = [
    MLPClassifier(hidden_layer_sizes=(7,), max_iter=300, batch_size=4, activation='logistic'),
    KNeighborsClassifier(n_neighbors=5),
    RandomForestClassifier(criterion='gini', max_depth=18, n_estimators=8),
    # AdaBoostClassifier(),
    DecisionTreeClassifier(max_depth=10),
    SVC(kernel="rbf"),
]


if __name__ == '__main__':
    x, y, data = load_transport_dataset()
    plot_pca_results(x, y)
    for model in classifiers:
         scores = cross_val_score(model, x, y, cv=10)
         print(model.__class__.__name__, ": ", scores)
    plot_correlation_matrix(data)
