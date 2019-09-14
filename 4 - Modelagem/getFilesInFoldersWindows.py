import os

import pandas as pd

dir_name = 'D:\\Gabriel\\PythonProjects\\epitopos\\MASTERMoveletsPivots\\epitopos_without_bepipred_ED\\test'  # Path to classes directory
os.chdir(dir_name)


def do_train():
    print('Generating Train clustering')
    directories = os.listdir(dir_name)
    print('Directories: {}'.format(directories))
    print('Extracting from directory named: {}'.format(directories[0]))
    dataset_train = pd.read_csv(directories[0] + '\\train.csv')
    labels = dataset_train[['class']]
    dataset_train = dataset_train[dataset_train.columns[:-1]]
    directories.pop(0)

    for directory in directories:
        new_folder = dir_name + '\\' + str(directory) + '\\'
        if os.path.isdir(new_folder):
            print('Extracting from directory named: {}'.format(directory))
            dataset_aux_train = pd.read_csv(directory + '\\train.csv')
            dataset_aux_train = dataset_aux_train[dataset_aux_train.columns[:-1]]
            dataset_train = dataset_train.join(dataset_aux_train)

    dataset_train = dataset_train.join(labels)
    dataset_train.to_csv(dir_name + '\\train.csv', index=None, header=True)
    print('\033[92m[OK] Train Clustering generated\033[0m')


def do_test():
    print('\nGenerating Train clustering')
    directories = os.listdir(dir_name)
    directories.remove('train.csv')

    print('Directories: {}'.format(directories))
    print('Extracting from directory named: {}'.format(directories[0]))
    dataset_test = pd.read_csv(directories[0] + '\\test.csv')
    labels = dataset_test[['class']]
    dataset_test = dataset_test[dataset_test.columns[:-1]]
    directories.pop(0)

    for directory in directories:
        new_folder = dir_name + '\\' + str(directory) + '\\'
        if os.path.isdir(new_folder):
            print('Extracting from directory named: {}'.format(directory))
            dataset_aux_test = pd.read_csv(directory + '\\test.csv')
            dataset_aux_test = dataset_aux_test[dataset_aux_test.columns[:-1]]
            dataset_test = dataset_test.join(dataset_aux_test)

    dataset_test = dataset_test.join(labels)
    dataset_test.to_csv(dir_name + '\\test.csv', index=None, header=True)
    print('\033[92m[OK] Test Clustering generated\033[0m')


do_train()
do_test()
