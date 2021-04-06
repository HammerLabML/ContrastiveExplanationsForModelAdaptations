#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import random

random.seed(42423)   # Credit amount, Present_residence_since and age become important 
np.random.seed(42423)

import pandas as pd
from copy import deepcopy
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier

from ceml.sklearn import generate_counterfactual
from utils import plot_feature_diff


if __name__ == "__main__":
    # Load data
    data = np.load("data.npz")
    X, y, features_desc = data["X"], data["y"], data["features_desc"]
    #print(features_desc)
    """
    ['Duration_in_month' 'Credit_amount'
 'Installment_rate_in_percentage_of_disposable_income'
 'Present_residence_since' 'Age_in_years'
 'Number_of_existing_credits_at_this_bank'
 'Number_of_people_being_liable_to_provide_maintenance_for'
 'Status_of_existing_checking_account_A11'
 'Status_of_existing_checking_account_A12'
 'Status_of_existing_checking_account_A13'
 'Status_of_existing_checking_account_A14' 'Credit_history_A30'
 'Credit_history_A31' 'Credit_history_A32' 'Credit_history_A33'
 'Credit_history_A34' 'Purpose_A40' 'Purpose_A41' 'Purpose_A410'
 'Purpose_A42' 'Purpose_A43' 'Purpose_A44' 'Purpose_A45' 'Purpose_A46'
 'Purpose_A48' 'Purpose_A49' 'Savings_account_bonds_A61'
 'Savings_account_bonds_A62' 'Savings_account_bonds_A63'
 'Savings_account_bonds_A64' 'Savings_account_bonds_A65'
 'Present_employment_since_A71' 'Present_employment_since_A72'
 'Present_employment_since_A73' 'Present_employment_since_A74'
 'Present_employment_since_A75' 'Personal_status_and_sex_A91'
 'Personal_status_and_sex_A92' 'Personal_status_and_sex_A93'
 'Personal_status_and_sex_A94' 'Other_debtors_guarantors_A101'
 'Other_debtors_guarantors_A102' 'Other_debtors_guarantors_A103'
 'Property_A121' 'Property_A122' 'Property_A123' 'Property_A124'
 'Other_installment_plans_A141' 'Other_installment_plans_A142'
 'Other_installment_plans_A143' 'Housing_A151' 'Housing_A152'
 'Housing_A153' 'Job_A171' 'Job_A172' 'Job_A173' 'Job_A174'
 'Telephone_A191' 'Telephone_A192' 'foreign_worker_A201'
 'foreign_worker_A202']
    """

    # Dimensionality reduction
    X = X[:, [0, 1, 2, 3, 4, 5, 6]]
    # y=0 <-> Reject
    # y=1 <-> Accept

    # Split data into two parts - age <= 35 and age > 35
    idx_young = X[:,4] <= 35
    idx_old = X[:,4] > 35
    X0, y0 = X[idx_young, :], y[idx_young]
    X1, y1 = X[idx_old, :], y[idx_old]

    # Split into train and test data
    X0_train, X0_test, y0_train, y0_test = train_test_split(X0, y0, test_size=0.2)
    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2)

    # Add artifical points for fixing counterfactuals - i.e. avoiding "problematic behaviour"
    X0_addon = []
    y0_addon = []
    for i in range(X0_train.shape[0]):
        if y0_train[i] == 0: # Reject
            x_new = deepcopy(X0_train[i,:])
            x_new[1] += 100 # Rejection must still hold if the applicant asks for more money!
            X0_addon.append(x_new)
            y0_addon.append(0)
    X0_addon = np.array(X0_addon)
    y0_addon = np.array(y0_addon)

    X1_addon = []
    y1_addon = []
    for i in range(X1_train.shape[0]):
        if y1_train[i] == 0: # Reject
            x_new = deepcopy(X1_train[i,:])
            x_new[1] += 100 # Rejection must still hold if the applicant asks for more money!
            X1_addon.append(x_new)
            y1_addon.append(0)
    X1_addon = np.array(X1_addon)
    y1_addon = np.array(y1_addon)

    #X0_train = np.concatenate((X0_train, X0_addon)) # Uncomment these lines to get the origina behaviour that asking for more money leads to an accept rather than a reject!
    #y0_train = np.concatenate((y0_train, y0_addon))
    #X1_train = np.concatenate((X1_train, X1_addon))
    #y1_train = np.concatenate((y1_train, y1_addon))

    # Create final sets
    Xcftest = np.concatenate((X0_test, X1_test))
    ycftest = np.concatenate((y0_test, y1_test))

    X_train = np.concatenate((X0_train, X1_train))
    y_train = np.concatenate((y0_train, y1_train))

    X_test = np.concatenate((X0_test, X1_test))
    y_test = np.concatenate((y0_test, y1_test))

    print(X0_train.shape, Xcftest.shape)

    # Fit a classifier
    model = DecisionTreeClassifier(max_depth=5)
    model.fit(X0_train, y0_train)
    print("Training: ROC-AUC-score: {0}".format(roc_auc_score(y0_train, model.predict(X0_train))))
    print("Test: ROC-AUC-score: {0}".format(roc_auc_score(y0_test, model.predict(X0_test))))
    print("Cf: ROC-AUC-score: {0}".format(roc_auc_score(ycftest, model.predict(Xcftest))))

    # Refit model using all data
    print("Refitting on new/all data....")
    model_old = deepcopy(model)
    model = DecisionTreeClassifier(max_depth=5)
    model.fit(X_train, y_train)
    print("Training: ROC-AUC-score: {0}".format(roc_auc_score(y_train, model.predict(X_train))))
    print("Test: ROC-AUC-score: {0}".format(roc_auc_score(y_test, model.predict(X_test))))
    print("Cf: ROC-AUC-score: {0}".format(roc_auc_score(ycftest, model.predict(Xcftest))))

    # Compute counterfactuals
    n_wrong_prediction = 0
    cf_new = []
    cf_old = []
    for i in range(Xcftest.shape[0]):
        x_orig = Xcftest[i]
        y_orig = ycftest[i]

        y_pred_old = model_old.predict([x_orig])
        y_pred = model.predict([x_orig])
        y_target = 1 - y_pred   # Target label: Flip
        if y_pred != y_orig or y_pred_old != y_orig:
            n_wrong_prediction += 1
            continue
    
        # Compute counterfactual under old model and under new model
        x_cf, _, delta_cf = generate_counterfactual(model_old, x_orig, y_target, return_as_dict=False)
        cf_old.append((x_orig, x_cf, y_target, delta_cf))
        
        x_cf, _, delta_cf = generate_counterfactual(model, x_orig, y_target, return_as_dict=False)
        cf_new.append((x_orig, x_cf, y_target, delta_cf))
    print(f"Wrong predictions: {n_wrong_prediction}/{Xcftest.shape[0]}")
    

    # Evaluation/Comparism of counterfactuals
    feature_diff_all = []
    feature_diff_0 = []
    feature_diff_1 = []
    for cf0, cf1 in zip(cf_old, cf_new):
        diff = cf0[3] - cf1[3]
        feature_diff_all.append(np.abs(diff))
        if cf0[2] == 0:
            feature_diff_0.append(diff)
        else:
            print(diff[1])
            feature_diff_1.append(diff)

    feature_diff_0 = np.array(feature_diff_0)
    feature_diff_1 = np.array(feature_diff_1)
    plot_feature_diff(feature_diff_0, title="Accept -> Reject", file_path_out="exp_results/credit_accept_reject_reg.pdf", xaxis_labels=['Duration_in_month', 'Credit_amount', 'Installment_rate', 'Present_residence_since', 'Age_in_years', 'Num_existing_credits', 'Num_people_maintenance'])
    plot_feature_diff(feature_diff_1, title="Reject -> Accept", file_path_out="exp_results/credit_reject_accept_reg.pdf", xaxis_labels=['Duration_in_month', 'Credit_amount', 'Installment_rate', 'Present_residence_since', 'Age_in_years', 'Num_existing_credits', 'Num_people_maintenance'])
    print(f"All: Mean diff: {np.mean(feature_diff_all,axis=0)}, Variance: {np.var(feature_diff_all, axis=0)}")
    print(f"1 -> 0: Mean diff: {np.mean(feature_diff_0,axis=0)}, Variance: {np.var(feature_diff_0, axis=0)}")  # Accept -> Reject
    print(f"0 -> 1 Mean diff: {np.mean(feature_diff_1,axis=0)}, Variance: {np.var(feature_diff_1, axis=0)}")  # Reject -> Accept