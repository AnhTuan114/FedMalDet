from typing import Tuple, Union, List
import numpy as np
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


XY = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[XY, XY]
LogRegParams = Union[XY, Tuple[np.ndarray]]
XYList = List[XY]

Params = Union[XY, Tuple[np.ndarray]]


def get_model_parameters(model: LogisticRegression) -> LogRegParams: #trả về tham số mô hình cục bộ
    """Returns the paramters of a sklearn LogisticRegression model."""
    if model.fit_intercept:
        params = [
            model.coef_,
            model.intercept_,
        ]
    else:
        params = [
            model.coef_,
        ]
    return params

def get_model_parameters1(model: LinearSVC) -> Params: #trả về tham số mô hình cục bộ
    """Returns the paramters of a sklearn model."""
    if model.fit_intercept:
        params = [
            model.coef_,
            model.intercept_,
        ]
    else:
        params = [
            model.coef_,
        ]
    return params

def set_model_params(       #khởi tạo tham số cho mô hình
    model: LogisticRegression, params: LogRegParams
) -> LogisticRegression:
    """Sets the parameters of a sklean LogisticRegression model."""
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model

def set_model_params1(       #khởi tạo tham số cho mô hình
    model: LinearSVC, params: Params
) -> LinearSVC:
    """Sets the parameters of a sklean LogisticRegression model."""
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model


def set_initial_params(model: LogisticRegression): #cập nhật tham số
    """Sets initial parameters as zeros Required since model params are
    uninitialized until model.fit is called.
    But server asks for initial parameters from clients at launch. Refer
    to sklearn.linear_model.LogisticRegression documentation for more
    information.
    """
    n_classes = 2  # MNIST has 10 classes
    n_features = 630  # Number of features in dataset
    model.classes_ = np.array([i for i in range(2)])

    model.coef_ = np.zeros((n_classes, n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((n_classes,))

def set_initial_params1(model: LinearSVC): #cập nhật tham số
    """Sets initial parameters as zeros Required since model params are
    uninitialized until model.fit is called.
    But server asks for initial parameters from clients at launch. Refer
    to sklearn.linear_model.LogisticRegression documentation for more
    information.
    """
    n_classes = 2  # MNIST has 10 classes
    n_features = 630  # Number of features in dataset
    model.classes_ = np.array([i for i in range(2)])
    model.coef_ = np.zeros((n_classes, n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((n_classes,))



def load_mnist() -> Dataset:
    df = pd.read_csv('D:\HỌC MÁY THỐNG KÊ\Project\Fed_Mal_Det\demo\REWEMA.csv')
    X = df.loc[:, '0':]
    y = df.loc[:, 'B']
    y = y.replace('B', 0).replace('M', 1)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return (x_train, y_train), (x_test, y_test)

def partition(X: np.ndarray, y: np.ndarray, num_partitions: int) -> XYList: #phân chia dữu liệu
    """Split X and y into a number of partitions."""
    return list(
        zip(np.array_split(X, num_partitions), np.array_split(y, num_partitions))
    )