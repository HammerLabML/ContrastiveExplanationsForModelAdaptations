# Contrastive Explanations for Explaining Model Adaptations

This repository contains the implementation of the methods proposed in the paper [Contrastive Explanations for Explaining Model Adaptations](paper.pdf) by André Artelt, Fabian Hinder, Valerie Vaquet, Robert Feldhans and Barbara Hammer -- note that the version in this paper is an extended version of the IWANN-2021 conference paper (e.g. more experiments, a new section on regularizing model adaptations, etc.).

The experiments as described in the paper are implemented in the folder [Code](Code/).

## Implementation of the different experiments

### Comparing Counterfactuals for Explaining Model Adaptation

#### Gaussian Blobs

Implemented in `Code/model_feature_drift.py` -- the script does not need any arguments.

#### Predicting House Prices

Implemented in `Code/model_drift_regression.py` -- the script does not need any arguments.

#### Coffee

Implemented in `Code/experiments_on_coffee.py` -- the script does not need any arguments.

Note that the data set can not be published.

#### Human Activity Recognition

Implemented in `Code/model_drift_har.py` -- the script does not need any arguments.

Note that the data set is to large to be uploaded.

#### Loan approval

Implementation can be found in `Code/model_drift_credit.py` -- the script does not need any arguments.



### Finding Relevant Regions in Data Space

#### Gaussian Blobs

Implemented in `Code/model_feature_drift.py` -- the script does not need any arguments.

In order to use the most important samples only, you have to comment out lines 69 and 87, uncomment lines 70 and 88 -- you might also want to change the path of the computed plot.

#### Human Activity Recognition

Implemented in `Code/model_drift_har.py` -- the script does not need any arguments.

In order to use the most important samples only, you have to comment out line 146, uncomment line 147 -- you might also want to change the path of the computed plot.


### Persistent Local Explanations

Implementation can be found in `Code/model_drift_credit.py` -- the script does not need any arguments. There is a global variable in the script `regularization` -- if `True`, a regularization for preventing unwanted changes in the internal reasoning is used (default is `False`).

## Data

All data sets are included, except the "Coffee data set" which can not be published because of copyright reasons, and the HAR data set which is too large to be uploaded (might be available upon request).

## Requirements

- Python3.6
- Packages as listed in `Code/REQUIREMENTS.txt`

## License

MIT license - See [LICENSE.md](LICENSE.md)

## How to cite

You can either cite the version on [arXiv](http://arxiv.org/abs/2104.02459) or the [IWANN-2021 conference version](https://link.springer.com/chapter/10.1007/978-3-030-85030-2_9).
