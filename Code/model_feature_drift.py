#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
random.seed(42)
import numpy as np
np.random.seed(42)
import cvxpy as cp
from numpy.random import multivariate_normal
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from copy import deepcopy

from ceml.sklearn import generate_counterfactual
from gnb_utils import compute_cf_gradient, compare_cf_gradients
from utils import plot_feature_diff


def sample_from_classdist(class_means, cov, n_samples_per_class=200):
    X = []
    y = []

    for i in range(class_means.shape[0]):
        y += [i for _ in range(n_samples_per_class)]
        for _ in range(n_samples_per_class):
            X.append(multivariate_normal(mean=class_means[i,:], cov=cov))

    X = np.array(X)
    y = np.array(y)

    return X, y


def generate_drifting_dataset():
    X0, y0 = sample_from_classdist(np.array([[0., 0.], [5., 0.]]), np.eye(2,2))
    X1, y1 = sample_from_classdist(np.array([[5., 5.]]), np.eye(2,2))

    return (X0, y0), (X1, y1)


if __name__ == "__main__":
    # Create data set:
    # Old data: Two blobs on the same axis - can be separated with a threshold on the first feature
    # New data: Another blob of the first class located approx. above the blob of the second class from the old data set => Second feature must be used when separating both classes!
    batches = generate_drifting_dataset()
    
    X0, y0 = batches[0]
    X1, y1 = batches[1]

    Xcftest, ycftest = sample_from_classdist(np.array([[0., 5.]]), np.eye(2,2))
    print(X0.shape, X1.shape, Xcftest.shape)
    # Test data for computing counterfactual explanations:
    # Blob above the blob of the first class from the old data set => Classification should still work fine since the second feature is not important for the first data set and the first feature remains important even after adapting to the new data set

    # Fit model on the first batch of data
    model = GaussianNB()
    model.partial_fit(X0, y0, classes=[0, 1])
    print("Accuracy: {0}".format(accuracy_score(y0, model.predict(X0))))

    # Adapt model to new second batch of data
    model_old = deepcopy(model)
    print("Accuracy on new data before adaptation: {0}".format(accuracy_score(y1, model.predict(X1))))
    model.partial_fit(X1, y1)
    print("Accuracy on new data after adaptation: {0}".format(accuracy_score(y1, model.predict(X1))))
    print("Accuracy after adaptation on old data: {0}".format(accuracy_score(y0, model.predict(X0))))

    # Find interesting samples
    cftest_scores = []
    for i in range(Xcftest.shape[0]):   # Only consider samples from a hold out set of samples
    #for i in cftest_scores_sorting[:10]:
        x = Xcftest[i,:]
        y_target = 0 if ycftest[i] == 1 else 1

        gradA = compute_cf_gradient(model_old, x, y_target)
        gradB = compute_cf_gradient(model, x, y_target)
        score = compare_cf_gradients(gradA, gradB)

        cftest_scores.append(score)
    cftest_scores_sorting = np.argsort(cftest_scores)

    # Compute counterfactuals under old and new model
    print("Accuracy on test data - old model: {0}".format(accuracy_score(ycftest, model_old.predict(Xcftest))))
    print("Accuracy on test data: {0}".format(accuracy_score(ycftest, model.predict(Xcftest))))

    cf_new = []
    cf_old = []
    for i in range(Xcftest.shape[0]):
    #for i in cftest_scores_sorting[:10]:
        x = Xcftest[i,:]
        y = ycftest[i]
        y_target = 0 if y == 1 else 1

        if model_old.predict([x]) != y or model.predict([x]) != y: # Check if both models classifiy the sample correctly!
            print("Skipping misslcassified sample!")
            continue

        x_cf, _, delta_cf = generate_counterfactual(model_old, x, y_target=y_target, return_as_dict=False, optimizer="mp", optimizer_args={"solver": cp.MOSEK})
        cf_old.append((x, x_cf, y_target, delta_cf))

        x_cf, _, delta_cf = generate_counterfactual(model, x, y_target=y_target, return_as_dict=False, optimizer="mp", optimizer_args={"solver": cp.MOSEK})
        cf_new.append((x, x_cf, y_target, delta_cf))

    # Compare counterfactuals
    feature_diff = []
    for cf0, cf1 in zip(cf_old, cf_new):
        diff = cf0[3] - cf1[3]
        feature_diff.append(diff)
    feature_diff = np.array(feature_diff)
    plot_feature_diff(feature_diff, file_path_out="exp_results/gaussianblobs_notrelevant.pdf")
    print(f"Mean difference: {np.mean(feature_diff,axis=0)}, Variance: {np.var(feature_diff, axis=0)}")
    # => Interpretation/Insight: The second feature becomes more important after adapting to the new data set! This is consistent with the ground truth :)
