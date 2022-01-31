#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from ceml.sklearn import generate_counterfactual
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import seaborn as sns

data_path = '/Datasets/Coffee_drift'


# parameters for experiments
date_t1 = '20200626'
set_up = ['correct-misclassified']


class_names = ['Arabica', 'Robusta', 'Immature']
cam = 'SWIR3170'
sampling_freq = 5

for date_t2 in ['20200814', '20200828']:

    # loading data at t1, training the model_t1
    dataset_t1 = np.load('{}/{}/{}_Train.npz'.format(data_path, date_t1, cam))
    dataset_test_t1 = np.load('{}/{}/{}_Test.npz'.format(data_path, date_t1, cam))
    data_t1 = dataset_t1['spectra']
    test_data_t1 = dataset_test_t1['spectra']
    labels_t1 = dataset_t1['labels'].astype(int).ravel()
    num_classes = len(np.unique(labels_t1))
    test_labels_t1 = dataset_test_t1['labels'].astype(int).ravel()

    wavelenghts = np.load('{}/wavelenghts_{}.npy'.format(data_path, cam))
    wavelenghts = np.round(wavelenghts[[i for i in range(0, len(wavelenghts), sampling_freq)]], decimals=1)

    data_t1 = data_t1[:, [i for i in range(0, data_t1.shape[1], sampling_freq)]]
    test_data_t1 = test_data_t1[:, [i for i in range(0, test_data_t1.shape[1], sampling_freq)]]


    model_t1 = LogisticRegression(multi_class='multinomial', max_iter=100)
    model_t1.fit(data_t1, labels_t1)
    print(accuracy_score(test_labels_t1, model_t1.predict(test_data_t1)))


    # loading data at t2, training the model_t2

    dataset_t2 = np.load('{}/{}/{}_Train.npz'.format(data_path, date_t2, cam))
    dataset_test_t2 = np.load('{}/{}/{}_Test.npz'.format(data_path, date_t2, cam))
    data_t2 = dataset_t2['spectra']
    test_data_t2 = dataset_test_t2['spectra']
    labels_t2 = dataset_t2['labels'].astype(int).ravel()
    test_labels_t2 = dataset_test_t2['labels'].astype(int).ravel()

    data_t2 = data_t2[:, [i for i in range(0, data_t2.shape[1], sampling_freq)]]
    test_data_t2 = test_data_t2[:, [i for i in range(0, test_data_t2.shape[1], sampling_freq)]]

    if zscore:
        data_t2 = stats.zscore(data_t2, ddof=1)
        test_data_t2 = stats.zscore(test_data_t2, ddof=1)

    model_t2 = LogisticRegression(multi_class='multinomial', max_iter=100)
    model_t2.fit(data_t2, labels_t2)
    print(accuracy_score(test_labels_t2, model_t2.predict(test_data_t2)))

    def get_counterfactual_samples(version):
        if version == 'correct-misclassified':
            pred_t1 = model_t1.predict(test_data_t1)
            results_t1 = pred_t1 == test_labels_t1
            pred_t2 = model_t2.predict(test_data_t1)
            results_t2 = pred_t2 != test_labels_t1
            indices = np.logical_and(results_t1, results_t2)
            print('found {} samples from time t1, that were correctly classfied by the model_t1 and misclassified by model_t2'.format(np.sum(indices)))
            cf_samples = test_data_t1[indices,:]
            cf_target_labels = test_labels_t1[indices]
            cf_target_labels_ = []
            model = model_t2
        elif version == 'misclassified-correct':
            pred_t1 = model_t1.predict(test_data_t1)
            results_t1 = pred_t1 != test_labels_t1
            pred_t2 = model_t2.predict(test_data_t1)
            results_t2 = pred_t2 == test_labels_t1
            indices = np.logical_and(results_t1, results_t2)
            print('found {} samples from time t1, that were misclassified by the model_t1 and correctly classfied by model_t2'.format(np.sum(indices)))
            cf_samples = test_data_t1[indices,:]
            cf_target_labels = model_t1.predict(cf_samples)
            cf_target_labels_ = []
            model = model_t2
        elif version == 'changes-not-affecting-classification':
            pred_t1 = model_t1.predict(test_data_t1)
            pred_t2 = model_t2.predict(test_data_t1)
            indices = pred_t1 == pred_t2
            print('found {} samples from time t1, that were classified the same way for both models'.format(np.sum(indices)))
            cf_samples = test_data_t1[indices,:]
            prediction = model_t1.predict(cf_samples)

            cf_target_labels = ((prediction + np.ones(len(prediction))) % 3).astype(int)
            cf_target_labels_ = ((prediction + np.ones(len(prediction))*2) % 3).astype(int)
            model = None
        elif version == 'changes-not-affecting-classification-correct':
            pred_t1 = model_t1.predict(test_data_t1)
            pred_t2 = model_t2.predict(test_data_t1)
            indices = pred_t1 == pred_t2
            indices_ = pred_t1 == labels_t1
            indices = np.logical_and(indices, indices_)
            print('found {} samples from time t1, that were classified correctly by both models'.format(np.sum(indices)))
            cf_samples = test_data_t1[indices,:]
            prediction = model_t1.predict(cf_samples)

            cf_target_labels = ((prediction + np.ones(len(prediction))) % 3).astype(int)
            cf_target_labels_ = ((prediction + np.ones(len(prediction))*2) % 3).astype(int)
            model = None

        return cf_samples, cf_target_labels, cf_target_labels_, model



    for version in set_up:
        print(version)
        for regularization in ['l1']:
            print(regularization)
            cf_samples, cf_target_labels, cf_target_labels_, model = get_counterfactual_samples(version)

            opt_args = {"solver_verbosity": False, "max_iter": 200}   # Use a better solving for gettign rid of solver errors ;)

            
            deltas = {orig : {target : [] for target in range(num_classes)} for orig in range(num_classes)}

            for i, y_target in enumerate(cf_target_labels):

                x_orig = cf_samples[i,:].ravel()

                if model is not None:
                    x_cf, y_cf, delta = generate_counterfactual(model, x_orig, y_target=y_target, return_as_dict=False, regularization=regularization, optimizer_args=opt_args)
                    deltas[model.predict(x_orig.reshape(1,-1))[0]][y_cf].append(delta)
                else:
                    x_cf, y_cf, delta = generate_counterfactual(model_t1, x_orig, y_target=y_target, return_as_dict=False, regularization=regularization, optimizer_args=opt_args)
                    x_cf_, y_cf_, delta_ = generate_counterfactual(model_t2, x_orig, y_target=y_target, return_as_dict=False, regularization=regularization, optimizer_args=opt_args)
                    deltas[model_t1.predict(x_orig.reshape(1,-1))[0]][y_cf].append(delta-delta_)

            for i, y_target in enumerate(cf_target_labels_):

                x_orig = cf_samples[i,:].ravel()

                if model is not None:
                    x_cf, y_cf, delta = generate_counterfactual(model, x_orig, y_target=y_target, return_as_dict=False, regularization=regularization, optimizer_args=opt_args)
                    deltas[model.predict(x_orig.reshape(1,-1))[0]][y_cf].append(delta)
                else:
                    x_cf, y_cf, delta = generate_counterfactual(model_t1, x_orig, y_target=y_target, return_as_dict=False, regularization=regularization, optimizer_args=opt_args)
                    x_cf_, y_cf_, delta_ = generate_counterfactual(model_t2, x_orig, y_target=y_target, return_as_dict=False, regularization=regularization, optimizer_args=opt_args)
                    deltas[model_t1.predict(x_orig.reshape(1,-1))[0]][y_cf].append(delta-delta_)

                
            for orig in range(num_classes):
                for target in range(num_classes):
                    if orig != target:
                        if len(deltas[orig][target])==0:
                            continue
                        df = pd.DataFrame(np.stack(deltas[orig][target], axis=1).T, columns=wavelenghts)
                        plt.figure()
                        plt.title('{}->{}'.format(class_names[orig], class_names[target]))
                        ax = sns.boxplot( data=df) 
                        plt.xticks(ticks=range(len(wavelenghts)), labels=wavelenghts, rotation=90, fontsize='xx-small')
                        plt.savefig('results/{}_{}_{}_boxplots_{}->{}_{}->{}.pdf'.format(version, regularization, zscore, date_t1, date_t2, orig, target), bbox_inches='tight')

                fig = plt.figure()
                for orig in range(num_classes):
                    for target in range(num_classes):
                        if orig != target:
                            if len(deltas[orig][target])==0:
                                continue

                            mean = np.mean(np.stack(deltas[orig][target], axis=1), axis=1)
                            var = np.var(np.stack(deltas[orig][target], axis=1), axis=1)
                            
                            np.savez('results/{}_{}_{}_{}_{}_mean_{}_{}.npz'.format(version, regularization, zscore, date_t1, date_t2, orig, target), deltas=deltas[orig][target], mean=mean, var=var)

                            plt.plot(wavelenghts, mean, label='{}->{}'.format(orig, target))
                            plt.close()
                        
                plt.legend()
                plt.savefig('results/{}_{}_{}_means_{}_{}.pdf'.format(version, regularization, zscore, date_t1, date_t2))
                plt.close()
                
    # parameters for experiments
    date_t1 = '20200626'
    date_t2 = '20200814'
    regularization = 'l1'

    cam = 'SWIR3170'
    sampling_freq = 5
    num_classes = 3
    colors = list(mcolors.TABLEAU_COLORS)

    wavelenghts = np.load('{}/wavelenghts_{}.npy'.format(data_path, cam))
    wavelenghts = np.round(wavelenghts[[i for i in range(0, len(wavelenghts), sampling_freq)]], decimals=1)
    for version in ['correct-misclassified', 'misclassified-correct', 'changes-not-affecting-classification']:
        for zscore in [True, False]:
        
            for regularization in ['l1', 'l2']:
                for date_t2 in ['20200814', '20200828']:
            
                    fig = plt.figure()
                    i=0
                    for orig in range(num_classes):
                        for target in range(num_classes):
                            if orig != target:
                                try:
                                    data = np.load('results/{}_{}_{}_{}_{}_mean_{}_{}.npz'.format(version, regularization, zscore, date_t1, date_t2, orig, target))
                                    mean = data['mean']
                                    var = data['var']
                                    X = data['deltas']
                                    median = np.median(X)
                                    plt.plot(wavelenghts, mean, c=colors[i], label='{}->{}'.format(orig, target))
                                    plt.plot(wavelenghts, mean+var, c=colors[i],alpha=0.4)
                                    plt.plot(wavelenghts, mean-var, c=colors[i],alpha=0.4)
                                    i+=1
                                except:
                                    print('results/{}_{}_{}_{}_{}_mean_{}_{}.npz not found'.format(version, regularization, zscore, date_t1, date_t2, orig, target))
                            
                    plt.ylim(-0.21, 0.155)
                    plt.legend()
                    if version == 'changes-not-affecting-classification':
                        plt.title('Difference in mean delta between models')
                    elif version == 'correct-misclassified':
                        plt.title('Mean delta for misclassification -> correct classification')
                    else:
                        plt.title('Mean delata misclassified -> correct')
                    plt.savefig('results/{}_{}_{}_means_var_{}_{}.pdf'.format(version, regularization, zscore, date_t1, date_t2))
                    plt.close()
