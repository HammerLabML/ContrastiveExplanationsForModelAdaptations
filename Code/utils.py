# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_feature_diff(diff, xaxis_labels=None, title="", file_path_out=None):
    data = {i+1: diff[:,i] for i in range(diff.shape[1])}
    df = pd.DataFrame(data)
    df = df.melt(var_name='Feature', value_name='Difference')

    plt.figure()
    plt.title(title)
    sns.boxplot(x='Feature', y='Difference', data=df)
    if xaxis_labels is not None:
        plt.xticks(ticks=range(diff.shape[1]), labels=xaxis_labels, rotation=90)
    if file_path_out is None:
        plt.show()
    else:
        plt.savefig(file_path_out, dpi=500, bbox_inches='tight', pad_inches=0)