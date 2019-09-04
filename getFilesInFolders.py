import os

import pandas as pd

dirr = "/usuarios/Analises_Colab_Glauber/Analises_Gabriel_S/Epitopos/AcademicsProjects/MASTERMoveletsPivots/epitopos_ED/Porcentage_10_Witout_Last_Prunning"
os.chdir(
    "/usuarios/Analises_Colab_Glauber/Analises_Gabriel_S/Epitopos/AcademicsProjects/MASTERMoveletsPivots/epitopos_ED/Porcentage_10_Witout_Last_Prunning")
directories = os.listdir(dirr)

print(directories)


def doTrain():
    dataset_train = pd.read_csv(directories[0] + "/train.csv")
    labels = dataset_train[['class']]
    dataset_train = dataset_train[dataset_train.columns[:-1]]
    directories.pop(0)

    for directory in directories:
        new_folder = dirr + '/' + str(directory) + '/'
        if (os.path.isdir(new_folder)):
            print(new_folder)
            dataset_aux_train = pd.read_csv(directory + "/train.csv")
            dataset_aux_train = dataset_aux_train[dataset_aux_train.columns[:-1]]
            dataset_train = dataset_train.join(dataset_aux_train)

    dataset_train = dataset_train.join(labels)
    dataset_train.to_csv(dirr + "/train.csv", index=None, header=True)


# dirr = 'D:/Camila Leite/Datasets/SurveyTests/MASTER Movelets/MASTERMovelets/Gowalla_ED'
# os.chdir('D:/Camila Leite/Datasets/SurveyTests/MASTER Movelets/MASTERMovelets/Gowalla_ED')

def doTest():
    print(directories)
    dataset_test = pd.read_csv(directories[0] + "/test.csv")
    print(directories[0])
    labels = dataset_test[['class']]
    dataset_test = dataset_test[dataset_test.columns[:-1]]
    directories.pop(0)

    for directory in directories:
        new_folder = dirr + '/' + str(directory) + '/'
        print(new_folder)
        if (os.path.isdir(new_folder)):
            print(new_folder)
            dataset_aux_test = pd.read_csv(directory + "/test.csv")
            dataset_aux_test = dataset_aux_test[dataset_aux_test.columns[:-1]]
            dataset_test = dataset_test.join(dataset_aux_test)

    dataset_test = dataset_test.join(labels)
    dataset_test.to_csv(dirr + "/test.csv", index=None, header=True)


doTest()
doTrain()
