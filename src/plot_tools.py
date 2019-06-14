""" Plot Tools

This script have a sets of tools that handle metrics and plots graphs, matrix and tables.
The tools here are implemented basically by Scikit, and adapted as demanded, bellow  is possible
see the references to this fonts:

- A compilation of bests 50 matplotlib visualization, to analysis data:
    https://www.machinelearningplus.com/plots/top-50-matplotlib-visualizations-the-master-plots-python/

- Implementations of Histograms, Density Plots, Box and Whisker Plots, Correlation Matrix Plot and Scatterplot Matrix:
    https://machinelearningmastery.com/visualize-machine-learning-data-python-pandas/

- API reference to Scikit metric:
    https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics

- API reference to Scikit plot metrics:
    https://scikit-plot.readthedocs.io/en/stable/metrics.html

This file can also be imported as a module and contains the following functions:
    * plot_confusion_matrix - plot/save and image of confusion matrix
    * get_metrics - return the metrics of a set of data
    * plot_metrics - show/save/print a table with metrics of a set of data
    * plot_metrics_graph - plot/save a graph of evolution of metrics in function of one variable
    * plot_distribution_data - plot a set of data in a 2D plan, showing the distribution,
      where axis x and y are 2 diff features
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statistics import mean
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import *
from scikitplot.metrics import *


def plot_confusion_matrix():
    print("do nothing")


def get_metrics():
    print("do nothing")


def plot_metrics():
    print("do nothing")


def plot_metrics_data():
    print("do nothing")


def plot_distribution_data():
    print("do nothing")

