'''
Created on May 16, 2018

@author: andres
'''
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

def Approach1(X_train, y_train, X_test, y_test, par_batch_size, par_epochs, par_lr, par_dropout, save_results, dir_path) :
    
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    from keras.optimizers import Adam
    import pandas as pd
    import os
        
    nattr = len(X_train[1,:])    
    
    # Scaling y and transforming to keras format
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    le.fit(y_train)
    y_train = le.transform(y_train) 
    y_test = le.transform(y_test)
    from keras.utils import to_categorical
    y_train1 = to_categorical(y_train)
    y_test1 = to_categorical(y_test)
    nclasses = len(le.classes_)
    
    #Initializing Neural Network
    classifier = Sequential()
    # Adding the input layer and the first hidden layer
    classifier.add(Dense(units = 100, kernel_initializer = 'uniform', activation = 'relu', input_dim = (nattr)))
    classifier.add(Dropout( par_dropout ))
    # Adding the output layer   
    classifier.add(Dense(units = nclasses, kernel_initializer = 'uniform', activation = 'softmax'))
    # Compiling Neural Network
    adam = Adam(lr=par_lr)
    classifier.compile(optimizer = adam, loss = 'categorical_crossentropy', metrics = ['accuracy','top_k_categorical_accuracy',f1])
    # Fitting our model 
    history = classifier.fit(X_train, y_train1, validation_data = (X_test, y_test1), batch_size = par_batch_size, epochs = par_epochs)
    # ---------------------------------------------------------------------------------
    
    # ---------------------------------------------------------------------------------
    if (save_results) :
        if not os.path.exists(dir_path + "/model/"):
            os.makedirs(dir_path + "/model/")
        classifier.save(dir_path + "/model/" + 'model_approach1.h5')
        from numpy import argmax
        y_test_true_dec = le.inverse_transform(argmax(y_test1, axis = 1))
        y_test_pred_dec =  le.inverse_transform(argmax( classifier.predict(X_test) , axis = 1)) 
        report = classification_report(y_test_true_dec, y_test_pred_dec )
        classification_report_csv(report, dir_path + "/model/" + 'model_approach1_report.csv',"Approach1")            
        pd.DataFrame(history.history).to_csv(dir_path + "/model/" + "model_approach1_history.csv")    
    
# --------------------------------------------------------------------------------------
def Approach2(X_train, y_train, X_test, y_test, par_batch_size, lst_par_epochs, lst_par_lr, par_dropout, save_results, dir_path) :
    
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    from keras.optimizers import Adam
    import pandas as pd
    import os
        
    nattr = len(X_train[1,:])    

    # Scaling y and transforming to keras format
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    le.fit(y_train)
    y_train = le.transform(y_train) 
    y_test = le.transform(y_test)
    from keras.utils import to_categorical
    y_train1 = to_categorical(y_train)
    y_test1 = to_categorical(y_test)
    nclasses = len(le.classes_)
    
    #Initializing Neural Network
    model = Sequential()
    # Adding the input layer and the first hidden layer
    model.add(Dense(units = 100, kernel_initializer = 'uniform', activation = 'relu', input_dim = (nattr)))
    model.add(Dropout( par_dropout ))
    # Adding the output layer
    model.add(Dense(units = nclasses, kernel_initializer = 'uniform', activation = 'softmax'))
    # Compiling Neural Network
    
    k = len(lst_par_epochs)
    
    for k in range(0,k) :
           
        adam = Adam(lr=lst_par_lr[k])
        model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy','top_k_categorical_accuracy',f1])
        history = model.fit(X_train, y_train1, validation_data = (X_test, y_test1), epochs=lst_par_epochs[k], batch_size=par_batch_size)
    
        # ---------------------------------------------------------------------------------
        if (save_results) :
            if not os.path.exists(dir_path + "/model/"):
                os.makedirs(dir_path + "/model/")
            model.save(dir_path + "/model/" + 'model_approach2_Step'+str(k+1)+'.h5')
            from numpy import argmax
            y_test_true_dec = le.inverse_transform(argmax(y_test1, axis = 1))
            y_test_pred_dec =  le.inverse_transform(argmax( model.predict(X_test) , axis = 1)) 
            report = classification_report(y_test_true_dec, y_test_pred_dec)
            classification_report_csv(report, dir_path + "/model/" + 'model_approach2_report_Step'+str(k+1)+'.csv', "Approach2_Step"+str(k+1)) 
            pd.DataFrame(history.history).to_csv(dir_path + "/model/" + 'model_approach2_history_Step'+str(k+1)+'.csv')  
    
    # ---------------------------------------------------------------------------------



def calculateAccTop5(classifier, X_test, y_test, K ):
    import numpy as np
    y_test_pred = classifier.predict_proba(X_test)
    order=np.argsort(y_test_pred, axis=1)
    n=classifier.classes_[order[:, -K:]]
    soma = 0;
    for i in range(0,len(y_test)) :
        if ( y_test[i] in n[i,:] ) :
            soma = soma + 1
    accTopK = soma / len(y_test)
    
    return accTopK

def ApproachSVMForThresholds(X_train, y_train, X_test, y_test, dir_path):
    
    from keras.models import Sequential
    import pandas as pd
    import os
    from sklearn import svm
    from sklearn.metrics import f1_score

    clf = svm.SVC(probability=True, kernel='linear', C=1, class_weight="balanced").fit(X_train, y_train)

    acc = clf.score(X_test,y_test)
    accTop5 = calculateAccTop5(clf, X_test, y_test, 5)  
    accTop1 = calculateAccTop5(clf, X_test, y_test, 1)  

    f_1 = f1_score(y_test, clf.predict(X_test), average='macro')  

    line=["ACC:", acc, "ACC@1:", accTop1, "ACC5:", accTop5, "F1:", f_1]
    # lines = list()
    # lines.append(line)
    # print(line)
    return acc

def ApproachSVM(X_train, y_train, X_test, y_test, save_results, dir_path):
    
    from keras.models import Sequential
    import pandas as pd
    import os
    from sklearn import svm
    from sklearn.metrics import f1_score

    clf = svm.SVC(probability=True, kernel='linear', C=1, class_weight="balanced").fit(X_train, y_train)

    acc = clf.score(X_test,y_test)
    accTop5 = calculateAccTop5(clf, X_test, y_test, 5)  
    accTop1 = calculateAccTop5(clf, X_test, y_test, 1)  

    f_1 = f1_score(y_test, clf.predict(X_test), average='macro')  

    line=["ACC:", acc, "ACC5:", accTop5, "F1:", f_1]
    lines = list()
    lines.append(line)

    print("My Results - SVM:", line)

    # ---------------------------------------------------------------------------------
    if (save_results) :
        if not os.path.exists(dir_path + "/model/"):
            os.makedirs(dir_path + "/model/")
        
        report = classification_report(y_test, clf.predict(X_test) )
        classification_report_csv(report, dir_path + "/model/" + "model_approachSVM_report.csv","SVM")        
        pd.DataFrame(lines).to_csv(dir_path + "/model/" + "model_approachSVM_history.csv") 


def ApproachRF(X_train, y_train, X_test, y_test, n_trees_set, save_results, dir_path) :
        
    import os
    import pandas as pd
    
    from sklearn.ensemble import RandomForestClassifier    
    from sklearn.metrics import f1_score
    # ---------------------------------------------------------------------------------
    
    lines = list()
    
    for n_tree in n_trees_set:
        classifier = RandomForestClassifier(verbose=0, n_estimators = n_tree, n_jobs = 8, random_state = 10)
        classifier.fit(X_train, y_train)
        acc = classifier.score(X_test,y_test)

        accTop5 = calculateAccTop5(classifier, X_test, y_test, 5)
        accTop1 = calculateAccTop5(classifier, X_test, y_test, 1)  

        f_1 = f1_score(y_test, classifier.predict(X_test), average='macro')  

        line=["Trees:", n_tree,"ACC:", acc, "ACC@1:", accTop1, "ACC5:", accTop5, "F1:", f_1]
        lines.append(line)
        print("My Results - RF:", line)
        
        # ---------------------------------------------------------------------------------
        if (save_results) :
            if not os.path.exists(dir_path + "/model/"):
                os.makedirs(dir_path + "/model/")
            
            report = classification_report(y_test, classifier.predict(X_test) )
            classification_report_csv(report, dir_path + "/model/" + "model_approachRF"+ str(n_tree) +"_report.csv","RF")        
            pd.DataFrame(lines).to_csv(dir_path + "/model/" + "model_approachRF"+ str(n_tree) +"_history.csv") 
    
    
    
def ApproachRFHP(X_train, y_train, X_test, y_test, save_results, dir_path) :
        
    import os
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import RandomizedSearchCV
        
    # ---------------------------------------------------------------------------------
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 100, stop = 500, num = 10)]    
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 100, num = 10)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
        
    rf = RandomForestClassifier(verbose=0, n_jobs = 8, random_state = 1, class_weight="balanced")
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 50, cv = 3, verbose=2, random_state=1, n_jobs = 1)
    rf_random.fit(X_train, y_train)
    
    print (rf_random.best_params_)
    
        
    classifier = rf_random.best_estimator_
    acc = classifier.score(X_test,y_test)
    accTop5 = calculateAccTop5(classifier, X_test, y_test, 5)
    line=[acc,accTop5]
    print(line)
        
    # ---------------------------------------------------------------------------------
    if (save_results) :
        if not os.path.exists(dir_path + "/model/"):
            os.makedirs(dir_path + "/model/")
        report = classification_report(y_test, classifier.predict(X_test) )
        classification_report_csv(report, dir_path + "/model/" + "model_approachRFHP_report.csv","DT") 
        pd.DataFrame(line).to_csv(dir_path + "/model/" + "model_approachRFHP_history.csv") 
    
# ----------------------------------------------------------------------------------

def ApproachDT(X_train, y_train, X_test, y_test, save_results, dir_path) :
        
    import os
    import pandas as pd
    
    from sklearn.tree import DecisionTreeClassifier    
    # ---------------------------------------------------------------------------------
    
    classifier = DecisionTreeClassifier()
    classifier.fit(X_train, y_train)
    acc = classifier.score(X_test,y_test)
    accTop5 = calculateAccTop5(classifier, X_test, y_test, 5)
    line=[acc, accTop5]
    print(line)
    
    # ---------------------------------------------------------------------------------
    if (save_results) :
        if not os.path.exists(dir_path + "/model/"):
            os.makedirs(dir_path + "/model/")
        report = classification_report(y_test, classifier.predict(X_test) )
        classification_report_csv(report, dir_path + "/model/" + "model_approachDT_report.csv","DT") 
        pd.DataFrame(line).to_csv(dir_path + "/model/" + "model_approachDT_history.csv") 
# ----------------------------------------------------------------------------------

def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

def ApproachMLP(X_train, y_train, X_test, y_test, par_batch_size, par_epochs, par_lr, par_dropout, save_results, dir_path) :
    
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    from keras.optimizers import Adam
    import pandas as pd
    import os
        
    nattr = len(X_train[1,:])    

    # Scaling y and transforming to keras format
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    le.fit(y_train)
    y_train = le.transform(y_train) 
    y_test = le.transform(y_test)
    from keras.utils import to_categorical
    y_train1 = to_categorical(y_train)
    y_test1 = to_categorical(y_test)
    nclasses = len(le.classes_)
    from keras import regularizers
    
    #Initializing Neural Network
    classifier = Sequential()
    # Adding the input layer and the first hidden layer
    classifier.add(Dense(units = 100, kernel_initializer = 'uniform', kernel_regularizer= regularizers.l2(0.02), input_dim = (nattr)))
    #classifier.add(BatchNormalization())
    classifier.add(Dropout(par_dropout))
    # Adding the output layer       
    classifier.add(Dense(units = nclasses, kernel_initializer = 'uniform', activation = 'softmax'))
    # Compiling Neural Network
    adam = Adam(lr=par_lr)
    classifier.compile(optimizer = adam, loss = 'categorical_crossentropy', metrics = ['accuracy',accuracy_like_sklearn, 'top_k_categorical_accuracy',f1])
    # Fitting our model 
    history = classifier.fit(X_train, y_train1, validation_data = (X_test, y_test1), batch_size = par_batch_size, epochs = par_epochs, verbose=0)
    
    print("My Results - MLP 198: acc:", history.history["val_acc"][197], ", acc5:", history.history["val_top_k_categorical_accuracy"][197], ", F1:", history.history["val_f1"][197])
    print("My Results - MLP 199: acc:", history.history["val_acc"][198], ", acc5:", history.history["val_top_k_categorical_accuracy"][198], ", F1:", history.history["val_f1"][198])
    print("My Results - MLP 200: acc:", history.history["val_acc"][199], ", acc5:", history.history["val_top_k_categorical_accuracy"][199], ", F1:", history.history["val_f1"][199])
    
    # ---------------------------------------------------------------------------------
    
    # ---------------------------------------------------------------------------------
    if (save_results) :
        if not os.path.exists(dir_path + "/model/"):
            os.makedirs(dir_path + "/model/")
        classifier.save(dir_path + "/model/" + 'model_MLP.h5')
        from numpy import argmax
        y_test_true_dec = le.inverse_transform(argmax(y_test1, axis = 1))
        y_test_pred_dec =  le.inverse_transform(argmax( classifier.predict(X_test) , axis = 1)) 
        report = classification_report(y_test_true_dec, y_test_pred_dec)

        Cnfsn_mtrx=confusion_matrix(y_test_true_dec, y_test_pred_dec)
        df_mtrx=(Cnfsn_mtrx,le.classes_)
        print(df_mtrx)
        classification_report_csv(report, dir_path + "/model/" + "model_approachMLP_report.csv","MLP") 
        pd.DataFrame(history.history).to_csv(dir_path + "/model/" + "model_MLP_history.csv")    
        

import keras.backend as K

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

from sklearn.metrics import accuracy_score
import numpy as np

def _process_pred(y_pred):
    argmax = np.argmax(y_pred, axis=1)
    y_pred = np.zeros(y_pred.shape)

    for row, col in enumerate(argmax):
        y_pred[row][col] = 1

    return y_pred

def accuracy_like_sklearn(y_true, y_pred):
    #accuracy_score(K.eval(y_true), K.eval(y_pred), normalize=True)
    return K.mean(y_pred)

# Importing the Keras libraries and packages (Verificar de onde foi pego este codigo
from keras import metrics

from keras import backend as K

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
# ----------------------------------------------------------------------------------


def classification_report_csv(report, reportfile, classifier):
    report_data = []
    lines = report.split('\n')   
    for line in lines[2:(len(lines)-3)]:
        row = {}        
        row_data = line.split()
        row['class'] = row_data[0]
        row['classifier'] = classifier
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        row['support'] = float(row_data[4])
        print(row)
        report_data.append(row)
    import pandas as pd
    dataframe = pd.DataFrame.from_dict(report_data)
    dataframe.to_csv(reportfile, index = False)


