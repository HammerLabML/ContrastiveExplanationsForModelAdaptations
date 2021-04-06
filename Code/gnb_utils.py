# -*- coding: utf-8 -*-
import numpy as np


def compute_cf_gradient(model, x, y):
    i = y   # y = target label
    j = 0 if y == 1 else 1  # Binary classification!

    A = np.diag(1. / model.sigma_[i, :]) - np.diag(1. / model.sigma_[j, :])
    b = (model.theta_[j, :] / model.sigma_[j, :]) - (model.theta_[i, :] / model.sigma_[i, :])

    grad = np.dot(A, x) + b

    return x - grad  # "Simple alternative": return grad


def compare_cf_gradients(gradA, gradB):
    return np.dot(gradA, gradB) / (np.linalg.norm(gradA)*np.linalg.norm(gradB)) # Cosine of the angle between the two cf gradients (vectors)