#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import cvxpy as cp
import logging
logging.disable(sys.maxsize) 
import random
random.seed(42)
import numpy as np
np.random.seed(42)
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston

from ceml.sklearn import generate_counterfactual
from utils import plot_feature_diff


if __name__ == "__main__":
    # Load data
    X, y = load_boston(return_X_y=True)

    epsilon = 5.

    # Spplit data into two parts - first part all houses where NOX is pretty low - second part all other houses with a 'higher' NOX value
    idx0 = X[:,4] <= 0.5
    idx1 = X[:,4] > 0.5

    X0, y0 = X[idx0, :], y[idx0]
    X1, y1 = X[idx1, :], y[idx1]

    # Split into train and test data
    X0_train, X0_test, y0_train, y0_test = train_test_split(X0, y0, test_size=0.2)
    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2)

    Xcftest = np.concatenate((X0_test, X1_test))
    ycftest = np.concatenate((y0_test, y1_test))

    print(X0_train.shape, Xcftest.shape)

    # Fit and evaluate model
    model = LinearRegression()
    model.fit(X0_train, y0_train)
    print(f"R^2: {r2_score(y0_test, model.predict(X0_test))}")
    print(f"R^2 on the second (unknown) batch: {r2_score(y1_test, model.predict(X1_test))}")

    # Compute counterfactual explanations
    cf_old = []
    for i in range(Xcftest.shape[0]):
        x = Xcftest[i,:]
        #print(model.predict(x.reshape(1,-1)), ycftest[i])
        y_target = 20.  # Arbitrary target (always the same) xD
        done = lambda z: np.abs(y_target - z) < epsilon

        try:
            x_cf, _, delta_cf = generate_counterfactual(model, x, y_target=y_target, done=done, return_as_dict=False, optimizer="mp", optimizer_args={"solver": cp.MOSEK})
            cf_old.append((x, x_cf, y_target, delta_cf))
        except Exception as ex:
            print(ex)
            cf_old.append((x, x, y_target, x - x))

    # Because many sklearn models do not support partial_fit :(
    X1_train = np.concatenate((X0_train, X1_train))
    y1_train = np.concatenate((y0_train, y1_train))
    X1_test = np.concatenate((X0_test, X1_test))
    y1_test = np.concatenate((y0_test, y1_test))

    print(f"R^2 on all data (before adaptation): {r2_score(y1_test, model.predict(X1_test))}")

    # Adapt model and evaluate it again
    model.fit(X1_train, y1_train)
    print(f"R^2 on all data: {r2_score(y1_test, model.predict(X1_test))}")

    # Compute counterfactuals of the same samples under the adapted model
    cf_new = []
    for i in range(Xcftest.shape[0]):
        x = Xcftest[i,:]
        y_target = 20.
        done = lambda z: np.abs(y_target - z) < epsilon

        try:
            x_cf, _, delta_cf = generate_counterfactual(model, x, y_target=y_target, done=done, return_as_dict=False, optimizer="mp", optimizer_args={"solver": cp.MOSEK})
            cf_new.append((x, x_cf, y_target, delta_cf))
        except:
            print("Error")
            cf_new.append((x, x, y_target, x - x))

    # Compare counterfactual explanations
    feature_diff = []
    for cf0, cf1 in zip(cf_old, cf_new):
        diff = cf0[3] - cf1[3]
        feature_diff.append(diff)
    feature_diff = np.array(feature_diff)
    #print(feature_diff)
    plot_feature_diff(feature_diff, xaxis_labels=["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"], file_path_out="exp_results/houseprices.pdf")
    print(f"Mean difference: {np.mean(feature_diff,axis=0)}, Variance: {np.var(feature_diff, axis=0)}")
    # => Interpretation: Feature 4 (NOX) becomes more important (changed) "a lot" - samples with higher NOX values were excluded in the beginning and introducted in the second batch. <- Inspect the weights of the linear regression model!