# coding: utf-8
import math
import operator
from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.pipeline

from sklearn import datasets
from sklearn.pipeline import Pipeline

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import clone

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from scipy.special import comb
