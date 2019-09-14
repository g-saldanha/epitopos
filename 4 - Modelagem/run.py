'''
Created on May 16, 2018

@author: andres
'''

# 
# Para garantir reprodutibilidade
from numpy.random import seed

seed(1)
from tensorflow import set_random_seed
from sklearn import preprocessing

set_random_seed(2)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Importing the libraries
import numpy as np
import pandas as pd
import time

# ---------------------------------------------------------------------------

dir_path = ""

startTime = time.time()

print("Loading train and test data from... " + dir_path)
dataset_train = pd.read_csv(dir_path + "/train.csv")
dataset_test = pd.read_csv(dir_path + "/test.csv")

nattr = len(dataset_train.iloc[1, :])
print("Number of attributes: " + str(nattr))

# print(dataset_train)
# print(dataset_test)
X_train = dataset_train.iloc[:, 0:(nattr - 1)].values
y_train = dataset_train.iloc[:, (nattr - 1)].values

X_test = dataset_test.iloc[:, 0:(nattr - 1)].values
y_test = dataset_test.iloc[:, (nattr - 1)].values

# Replace distance 0 for presence 1
# # and distance 2 to non presence 0
X_train[X_train == 0] = 1
X_train[X_train == 2] = 0
X_test[X_test == 0] = 1
X_test[X_test == 2] = 0

X_train = preprocessing.scale(X_train);   
X_test = preprocessing.scale(X_test);

min_max_scaler = preprocessing.MinMaxScaler()
#X_train = min_max_scaler.fit_transform(X_train)
#X_test = min_max_scaler.transform(X_test)

# ----------------------------------------------------------------------------------
from Methods import ApproachRF

par_droupout = 0.5
par_batch_size = 200
par_epochs = 200
par_lr = 0.00095
save_results = False

# ----------------------------------------------------------------------------------

n_estimators = np.arange(10, 751, 10)
# n_estimators = np.append([1], n_estimators)
n_estimators = [100]
print(n_estimators)

print("Dropout: ", par_droupout, ". Batch: ", par_batch_size, ". Epoch: ", par_epochs, ". Learning Rate: ", par_lr)
# ApproachMLP(X_train, y_train, X_test, y_test, par_batch_size, par_epochs, par_lr, par_droupout, save_results, dir_path)

#################################################################################

ApproachRF(X_train, y_train, X_test, y_test, n_estimators, save_results, dir_path)

# ApproachSVM(X_train, y_train, X_test, y_test, save_results, dir_path)

# ---------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------

elapsedTime = time.time() - startTime
print('Time spent:', int(elapsedTime * 1000))

# print("Done.")
print("Finished.")

# ---------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------
