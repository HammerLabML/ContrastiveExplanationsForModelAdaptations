#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import cvxpy as cp
import numpy as np
import logging
logging.disable(sys.maxsize) 
from copy import deepcopy

import random
random.seed(4242)
np.random.seed(4242)

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from ceml.sklearn import generate_counterfactual
from gnb_utils import compute_cf_gradient, compare_cf_gradients
from utils import plot_feature_diff



def create_features(X):
    phi = []
    row_count = X.shape[0]

    def compute_features(X):
        features = []
        col_count = X.shape[1]

        for i in range(0, col_count):
            col = X[:, i]
            features.append(np.median(col))

        return features

    for i in range(0, row_count):  # Process each sample
        # Compute features
        features = compute_features(X[i])
        phi.append(features)


    phi = np.array(phi).astype(np.float)  # To numpy array

    return phi


# Load Activitiy Recognition data set
def generate_drifting_har_data(data_path="/home/aartelt/Documents/ActivityRecognition/Dataset1"):
    subjects = []
    for i in range(1, 30):
        if i < 10:
            subjects.append(f"0{i}")
        else:
            subjects.append(f"{i}")

    X = []; y = []
    for s in subjects:
        data = np.load(os.path.join(data_path, f"{s}-segmented.npz"))
        X_, y_ = data["X"], data["y"]
        X_ = create_features(X_)

        X.append(X_);y.append(y_)
    X = np.concatenate(X)
    y = np.concatenate(y)
   
    # Select based on label (activity)
    idx0 = y == 0   # WALKING
    idx1 = y == 1  # WALKING_UPSTAIRS
    idx2 = y == 2  # WALKING_DOWNSTAIRS

    X_default = X[idx0, :]
    idx_default0 = range(int(X_default.shape[0] / 2));idx_default1 = range(int(X_default.shape[0] / 2), X_default.shape[0])
    X_upstairs = X[idx1, :]
    X_downstairs = X[idx2, :]

    X_default0 = X_default[idx_default0,:]
    X_default1 = X_default[idx_default1,:]

    X0 = np.concatenate((X_default0, X_upstairs))
    X1 = np.concatenate((X_default1, X_downstairs))
    y0 = np.concatenate((np.zeros(X_default0.shape[0]), np.ones(X_upstairs.shape[0])))
    y1 = np.concatenate((np.zeros(X_default1.shape[0]), np.ones(X_downstairs.shape[0])))

    return (X0, y0), (X1, y1)


if __name__ == "__main__":
    # Create data set
    batches = generate_drifting_har_data()
    
    X0, y0 = batches[0]
    X1, y1 = batches[1]
    print(X0.shape, X1.shape)

    # Split data into train and test set - only a few samples for test samples for comparing counterfactual explanations
    X0_train, X0_test, y0_train, y0_test = train_test_split(X0, y0, test_size=0.3)
    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.3)

    Xcftest = np.concatenate((X0_test, X1_test))
    ycftest = np.concatenate((y0_test, y1_test))
    print(Xcftest.shape)

    # Fit model on the first batch of data
    model = GaussianNB()
    model.partial_fit(X0_train, y0_train, classes=[0, 1])
    print("Accuracy: {0}".format(accuracy_score(y0_test, model.predict(X0_test))))
    print("Accuracy on counterfactual test samples: {0}".format(accuracy_score(ycftest, model.predict(Xcftest))))

    # Adapt model to new second batch of data
    model_old = deepcopy(model)
    print("Accuracy on new data before adaptation: {0}".format(accuracy_score(y1_test, model.predict(X1_test))))
    model.partial_fit(X1_train, y1_train)
    print("Accuracy on new data after adaptation: {0}".format(accuracy_score(y1_test, model.predict(X1_test))))
    print("Accuracy after adaptation on old data: {0}".format(accuracy_score(y0_test, model.predict(X0_test))))
    print("Accuracy on counterfactual test samples: {0}".format(accuracy_score(ycftest, model.predict(Xcftest))))

    # Find interesting samples
    cftest_scores = []
    for i in range(Xcftest.shape[0]):   # Only consider samples from a hold out set of samples
        x = Xcftest[i,:]
        y_target = 0 if ycftest[i] == 1 else 1

        gradA = compute_cf_gradient(model_old, x, y_target)
        gradB = compute_cf_gradient(model, x, y_target)
        score = compare_cf_gradients(gradA, gradB)

        cftest_scores.append(score))
    cftest_scores_sorting = np.argsort(cftest_scores)

    # Compute counterfactuals under old and new model
    print("Accuracy on test data - old model: {0}".format(accuracy_score(ycftest, model_old.predict(Xcftest))))
    print("Accuracy on test data: {0}".format(accuracy_score(ycftest, model.predict(Xcftest))))

    cf_new = []
    cf_old = []
    #for i in range(Xcftest.shape[0]):
    for i in cftest_scores_sorting[500:]:
        x = Xcftest[i,:]
        y = ycftest[i]
        y_target = 0 if y == 1 else 1

        if model_old.predict([x]) != y or model.predict([x]) != y: # Check if both models classifiy the sample correctly!
            print(f"Skipping misslcassified sample {i}/{Xcftest.shape[0]}")
            continue

        x_cf, _, delta_cf = generate_counterfactual(model_old, x, y_target=y_target, return_as_dict=False, optimizer="mp", optimizer_args={"solver": cp.MOSEK})
        cf_old.append((x, x_cf, y_target, delta_cf))

        x_cf, _, delta_cf = generate_counterfactual(model, x, y_target=y_target, return_as_dict=False, optimizer="mp", optimizer_args={"solver": cp.MOSEK})
        cf_new.append((x, x_cf, y_target, delta_cf))


    # Compare counterfactual explanations
    feature_diff_all = []
    feature_diff_0 = []
    feature_diff_1 = []
    for cf0, cf1 in zip(cf_old, cf_new):
        diff = cf0[3] - cf1[3]
        feature_diff_all.append(np.abs(diff))
        if cf0[2] == 0:
            feature_diff_0.append(diff)
        else:
            feature_diff_1.append(diff)
    feature_diff_0 = np.array(feature_diff_0)
    feature_diff_1 = np.array(feature_diff_1)

    plot_feature_diff(feature_diff_0, title="Stairs -> Walking", xaxis_labels=["X Acc", "Y Acc", "Z Acc", "X Gyro", "Y Gyro", "Z Gyro"], file_path_out="exp_results/har_stairs_walking_notrelevant.pdf")
    plot_feature_diff(feature_diff_1, title="Walking -> Stairs", xaxis_labels=["X Acc", "Y Acc", "Z Acc", "X Gyro", "Y Gyro", "Z Gyro"], file_path_out="exp_results/har_walking_stairs_notrelevant.pdf")

    print(f"All: Mean difference: {np.mean(feature_diff_all,axis=0)}, Variance: {np.var(feature_diff_all, axis=0)}")
    print(f"1 -> 0: Mean difference: {np.mean(feature_diff_0,axis=0)}, Variance: {np.var(feature_diff_0, axis=0)}")
    print(f"0 -> 1 Mean difference: {np.mean(feature_diff_1,axis=0)}, Variance: {np.var(feature_diff_1, axis=0)}")
    # Interpretation: First three features are acceleration sensors (one for each axis) and the last three features decode the gyro sensors (one for each axis)
    # Observation: Importance of X and Y axis are swapped between the two target labels - The importance of X and mostly Y axis (acc) and Y axis (gyro) seems to change after model drift
    # When going from stairs to walking: X axis becomes more important but the importance of the Y axis decreases a lot - holds for acc and gyro sensors
    # When going from walking to stairs: X axis becomes less important but the importance of the Y axis increases - only holds for acc, gyro sensor does not change
    # Makes sense -> Switching between (adapting) walking upstairs and downstairs should change Y axis - instead of relying on Y axis only, the x axis becomes also important
