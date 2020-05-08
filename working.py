# -*- coding: utf-8 -*-
"""
Created on Fri May  8 02:03:50 2020

@author: jethi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn


data=pd.read_csv("data.csv")
X= data.iloc[:,:-1].values
Y= data.iloc[:,3].values

""" missing values- imputer """

from sklearn.impute import SimpleImputer
imputer= SimpleImputer(missing_values=np.nan, strategy='mean',verbose=0)
imputer.fit(X[:,1:3])
X[:,1:3]= imputer.transform(X[:,1:3])



