#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 16:15:03 2018

@author: julio
"""
import itertools
import pandas as pd
from pandas import read_csv
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.decomposition import PCA
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


pd.options.display.max_columns=99

dataset=read_csv('./Mushroom/mushroom.csv',header=0)



#describimos y vemos lo que hay. Vemos missing values
print(dataset.describe())
print(len(dataset.index))
print(len(dataset.columns))

#print((dataset['cap-shape'].value_counts()))

#print((dataset[['cap-shape']]=="NaN").sum())

#print(dataset['cap-shape'].count())


#cambiamos missing values por NaN

dataset.replace(r'\s+', np.NaN, regex=True)

#eliminamos misssing values
dataset.dropna(inplace=True)
plt.figure(figsize=(16,16))
sns.heatmap(dataset.apply(lambda x : pd.factorize(x)[0]).corr(method='pearson', min_periods=1), annot=True, cmap="Blues")
plt.show()


#Otra opcion: cambiamos por el mas aparecido
"""
dataset = dataset.apply(lambda x:x.fillna(x.value_counts().index[0]))
"""

#vemos lo que hay (se han eliminado 170 columnas)
print(dataset.describe())

#hacemos una correlación







#veil-type lo podemos obviar
print(dataset['veil-type'].describe())

dataset.drop(['veil-type'],axis=1,inplace=True)




class_count = dataset['Class'].value_counts()
sns.set(style="darkgrid")
sns.barplot(class_count.index, class_count.values, alpha=0.9)
plt.title('Frequency Distribution of Classes')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Class', fontsize=12)
plt.show()



#One hot coding

cont = 1
for col in ['cap-shape','cap-surface','cap-color', 'bruises', 'odor', 'gill-attachment', 'gill-spacing','gill-size','gill-color','stalk-shape','stalk-surface-above-ring','stalk-surface-below-ring','stalk-color-above-ring','stalk-color-below-ring','veil-color','ring-number','ring-type','spore-print-color','population','habitat']:
    dummy_col = pd.get_dummies(dataset[col],prefix='f{}'.format(cont))
    dataset = pd.concat([dataset, dummy_col], axis=1)
    dataset.drop(col, axis=1,inplace=True)
    cont += 1

dataset['Class'] = dataset['Class'].map({'e':1, 'p':0})

print(dataset.head())

X = dataset.drop('Class',1)
Y = dataset.Class


print(X.head(5))

#How classes are distributed
kf = KFold(n_splits=5)
    
  
for train_index, test_index in kf.split(X):
        
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
    print(Y_test.head(5))
    print(Y_test.value_counts())
    
     

#feature selection classific
"""
select = SelectKBest(chi2,k=10)
selected_features = select.fit(X_train, Y_train)
indices_selected = selected_features.get_support(indices=True)
colnames_selected = [X.columns[i] for i in indices_selected]

X_train = X_train[colnames_selected]
X_test = X_test[colnames_selected]

print(colnames_selected)
"""





#dimensionality reduction




#Code to print confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
   

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

np.set_printoptions(precision=2)
plt.figure()

# simple lineal regression PCA
"""
def lineal_regression_PCA(X, Y):
    pca = PCA(n_components=10)
    pca.fit(X)
    T = pca.transform(X)
    T= pd.DataFrame(T)                   

    print(T.head())
    
    print('\n')
    print('\n')

    kf = KFold(n_splits=5)
    auc = []
    cnf_matrix = []
  
    for train_index, test_index in kf.split(T):
        
        X_train, X_test = T.iloc[train_index], T.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        
       
        model = LogisticRegression()
        model.fit(X_train, Y_train)
        y_hat = [x for x in model.predict(X_test)]
        auc.append(accuracy_score(Y_test, y_hat))
        
        cnf_matrix.append(confusion_matrix(Y_test, y_hat))  
       
    
    X_train, X_test, Y_train, Y_test = train_test_split(T,Y,train_size=0.8,random_state=15)
    model = LogisticRegression()
    model.fit(X_train, Y_train)
    y_hat = [x for x in model.predict(X_test)]
    auc.append(accuracy_score(Y_test, y_hat))
    cnf_matrix.append(confusion_matrix(Y_test, y_hat))  
   
    plt.figure()
    plot_confusion_matrix(cnf_matrix[0], classes=['Edible1','Poisonous1'],title='Confusion Matrix')
    plt.figure()
    plot_confusion_matrix(cnf_matrix[1], classes=['Edible2','Poisonous2'],title='Confusion Matrix')
    plt.figure()
    plot_confusion_matrix(cnf_matrix[2], classes=['Edible2','Poisonous3'],title='Confusion Matrix')
    plt.figure()
    plot_confusion_matrix(cnf_matrix[3], classes=['Edible4','Poisonous4'],title='Confusion Matrix')
    plt.figure()
    plot_confusion_matrix(cnf_matrix[4], classes=['Edible5','Poisonous5'],title='Confusion Matrix')
    plt.figure()
    plot_confusion_matrix(cnf_matrix[5], classes=['Edible6','Poisonous6'],title='Confusion Matrix')
    print('Acuraccy array: {auc}'.format(auc=auc))
    average = 0
    for x in auc:
        average += x
        
    average = average/6
    print('Accuracy average: {average}'.format(average=average))
    return auc

auc_lineal_regression_PCA = lineal_regression_PCA(X,Y)
"""

# simple lineal regression SelectKBest
"""
def lineal_regression_KBest(X, Y):
    
    
    print('\n')
    print('\n')

    kf = KFold(n_splits=5)
    auc = []
    cnf_matrix = []
  
    for train_index, test_index in kf.split(X):
        
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        
        select = SelectKBest(chi2,k=10)
        selected_features = select.fit(X_train, Y_train)
        indices_selected = selected_features.get_support(indices=True)
        colnames_selected = [X.columns[i] for i in indices_selected]

        X_train = X_train[colnames_selected]
        X_test = X_test[colnames_selected]

        
       
        model = LogisticRegression()
        model.fit(X_train, Y_train)
        y_hat = [x for x in model.predict(X_test)]
        auc.append(accuracy_score(Y_test, y_hat))

        cnf_matrix.append(confusion_matrix(Y_test, y_hat))  
     
    
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,train_size=0.8,random_state=15)
    select = SelectKBest(chi2,k=10)
    selected_features = select.fit(X_train, Y_train)
    indices_selected = selected_features.get_support(indices=True)
    colnames_selected = [X.columns[i] for i in indices_selected]

    X_train = X_train[colnames_selected]
    X_test = X_test[colnames_selected]
        
    model = LogisticRegression()
    model.fit(X_train, Y_train)
    y_hat = [x for x in model.predict(X_test)]
    auc.append(accuracy_score(Y_test, y_hat))
    cnf_matrix.append(confusion_matrix(Y_test, y_hat))  
    
    
    plt.figure()
    plot_confusion_matrix(cnf_matrix[0], classes=['Edible1','Poisonous1'],title='Confusion Matrix')
    plt.figure()
    plot_confusion_matrix(cnf_matrix[1], classes=['Edible2','Poisonous2'],title='Confusion Matrix')
    plt.figure()
    plot_confusion_matrix(cnf_matrix[2], classes=['Edible3','Poisonous3'],title='Confusion Matrix')
    plt.figure()
    plot_confusion_matrix(cnf_matrix[3], classes=['Edible4','Poisonous4'],title='Confusion Matrix')
    plt.figure()
    plot_confusion_matrix(cnf_matrix[4], classes=['Edible5','Poisonous5'],title='Confusion Matrix')
    plt.figure()
    plot_confusion_matrix(cnf_matrix[5], classes=['Edible6','Poisonous6'],title='Confusion Matrix')
    print('Acuraccy array: {auc}'.format(auc=auc))
    average = 0
    for x in auc:
        average += x
        
    average = average/6
    print('Accuracy average: {average}'.format(average=average))
    return auc

auc_lineal_regression_KBest = lineal_regression_KBest(X,Y)
"""

# simple lineal regression ExtraTrees
"""
def lineal_regression_ExtraTrees(X, Y):
    
    
    print('\n')
    print('\n')

    kf = KFold(n_splits=5)
    auc = []
    cnf_matrix = []
  
    for train_index, test_index in kf.split(X):
        
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        
        select = ExtraTreesClassifier()
        select = select.fit(X_train, Y_train)
        trees = SelectFromModel(select, prefit=True)
        X_train = trees.transform(X_train)
        X_test = trees.transform(X_test)

        
       
        model = LogisticRegression()
        model.fit(X_train, Y_train)
        y_hat = [x for x in model.predict(X_test)]
        auc.append(accuracy_score(Y_test, y_hat))

        cnf_matrix.append(confusion_matrix(Y_test, y_hat))  
     
    
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,train_size=0.8,random_state=15)
    select = ExtraTreesClassifier()
    select = select.fit(X_train, Y_train)
    trees = SelectFromModel(select, prefit=True)
    X_train = trees.transform(X_train)
    X_test = trees.transform(X_test)
    
   
    
    
        
    model = LogisticRegression()
    model.fit(X_train, Y_train)
    y_hat = [x for x in model.predict(X_test)]
    auc.append(accuracy_score(Y_test, y_hat))
    cnf_matrix.append(confusion_matrix(Y_test, y_hat))  
    
    
    plt.figure()
    plot_confusion_matrix(cnf_matrix[0], classes=['Edible1','Poisonous1'],title='Confusion Matrix')
    plt.figure()
    plot_confusion_matrix(cnf_matrix[1], classes=['Edible2','Poisonous2'],title='Confusion Matrix')
    plt.figure()
    plot_confusion_matrix(cnf_matrix[2], classes=['Edible3','Poisonous3'],title='Confusion Matrix')
    plt.figure()
    plot_confusion_matrix(cnf_matrix[3], classes=['Edible4','Poisonous4'],title='Confusion Matrix')
    plt.figure()
    plot_confusion_matrix(cnf_matrix[4], classes=['Edible5','Poisonous5'],title='Confusion Matrix')
    plt.figure()
    plot_confusion_matrix(cnf_matrix[5], classes=['Edible6','Poisonous6'],title='Confusion Matrix')
    print('Acuraccy array: {auc}'.format(auc=auc))
    average = 0
    for x in auc:
        average += x
        
    average = average/6
    print('Accuracy average: {average}'.format(average=average))
    return auc

auc_lineal_regression_KBest = lineal_regression_KBest(X,Y)
"""
# SVM selectKbest
"""
def svm_KBest(X, Y):
    
    
    print('\n')
    print('\n')

    kf = KFold(n_splits=5)
    auc = []
    cnf_matrix = []
  
    for train_index, test_index in kf.split(X):
        
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        
        select = SelectKBest(chi2,k=10)
        selected_features = select.fit(X_train, Y_train)
        indices_selected = selected_features.get_support(indices=True)
        colnames_selected = [X.columns[i] for i in indices_selected]

        X_train = X_train[colnames_selected]
        X_test = X_test[colnames_selected]

        
       
        model = SVC(kernel='linear')
        model.fit(X_train, Y_train)
        y_hat = [x for x in model.predict(X_test)]
        auc.append(accuracy_score(Y_test, y_hat))

        cnf_matrix.append(confusion_matrix(Y_test, y_hat))  
     
    
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,train_size=0.8,random_state=15)
    select = SelectKBest(chi2,k=10)
    selected_features = select.fit(X_train, Y_train)
    indices_selected = selected_features.get_support(indices=True)
    colnames_selected = [X.columns[i] for i in indices_selected]

    X_train = X_train[colnames_selected]
    X_test = X_test[colnames_selected]
        
    model = SVC(kernel='linear')
    model.fit(X_train, Y_train)
    y_hat = [x for x in model.predict(X_test)]
    auc.append(accuracy_score(Y_test, y_hat))
    cnf_matrix.append(confusion_matrix(Y_test, y_hat))  
    
    
    plt.figure()
    plot_confusion_matrix(cnf_matrix[0], classes=['Edible1','Poisonous1'],title='Confusion Matrix')
    plt.figure()
    plot_confusion_matrix(cnf_matrix[1], classes=['Edible2','Poisonous2'],title='Confusion Matrix')
    plt.figure()
    plot_confusion_matrix(cnf_matrix[2], classes=['Edible3','Poisonous3'],title='Confusion Matrix')
    plt.figure()
    plot_confusion_matrix(cnf_matrix[3], classes=['Edible4','Poisonous4'],title='Confusion Matrix')
    plt.figure()
    plot_confusion_matrix(cnf_matrix[4], classes=['Edible5','Poisonous5'],title='Confusion Matrix')
    plt.figure()
    plot_confusion_matrix(cnf_matrix[5], classes=['Edible6','Poisonous6'],title='Confusion Matrix')
    print('Acuraccy array: {auc}'.format(auc=auc))
    average = 0
    for x in auc:
        average += x
        
    average = average/6
    print('Accuracy average: {average}'.format(average=average))
    return auc

auc_svm_KBest = svm_KBest(X,Y)
"""
#SVM polynomial kernel selectKbest
"""
def svm_polynomial_KBest(X, Y):
    
    
    print('\n')
    print('\n')

    kf = KFold(n_splits=5)
    auc = []
    cnf_matrix = []
  
    for train_index, test_index in kf.split(X):
        
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        
        select = SelectKBest(chi2,k=10)
        selected_features = select.fit(X_train, Y_train)
        indices_selected = selected_features.get_support(indices=True)
        colnames_selected = [X.columns[i] for i in indices_selected]

        X_train = X_train[colnames_selected]
        X_test = X_test[colnames_selected]

        
       
        model = SVC(kernel='poly', degree=7)
        model.fit(X_train, Y_train)
        y_hat = [x for x in model.predict(X_test)]
        auc.append(accuracy_score(Y_test, y_hat))

        cnf_matrix.append(confusion_matrix(Y_test, y_hat))  
     
    
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,train_size=0.8,random_state=15)
    select = SelectKBest(chi2,k=10)
    selected_features = select.fit(X_train, Y_train)
    indices_selected = selected_features.get_support(indices=True)
    colnames_selected = [X.columns[i] for i in indices_selected]

    X_train = X_train[colnames_selected]
    X_test = X_test[colnames_selected]
        
    model = SVC(kernel='poly', degree=2)
    model.fit(X_train, Y_train)
    y_hat = [x for x in model.predict(X_test)]
    auc.append(accuracy_score(Y_test, y_hat))
    cnf_matrix.append(confusion_matrix(Y_test, y_hat))  
    
    
    plt.figure()
    plot_confusion_matrix(cnf_matrix[0], classes=['Edible1','Poisonous1'],title='Confusion Matrix')
    plt.figure()
    plot_confusion_matrix(cnf_matrix[1], classes=['Edible2','Poisonous2'],title='Confusion Matrix')
    plt.figure()
    plot_confusion_matrix(cnf_matrix[2], classes=['Edible3','Poisonous3'],title='Confusion Matrix')
    plt.figure()
    plot_confusion_matrix(cnf_matrix[3], classes=['Edible4','Poisonous4'],title='Confusion Matrix')
    plt.figure()
    plot_confusion_matrix(cnf_matrix[4], classes=['Edible5','Poisonous5'],title='Confusion Matrix')
    plt.figure()
    plot_confusion_matrix(cnf_matrix[5], classes=['Edible6','Poisonous6'],title='Confusion Matrix')
    print('Acuraccy array: {auc}'.format(auc=auc))
    average = 0
    for x in auc:
        average += x
        
    average = average/6
    print('Accuracy average: {average}'.format(average=average))
    return auc

auc_svm_polynomial_KBest = svm_KBest(X,Y)
"""

#SVM gaussian kernel selectKbest
"""
def svm_gaussian_KBest(X, Y):
    
    
    print('\n')
    print('\n')

    kf = KFold(n_splits=5)
    auc = []
    cnf_matrix = []
  
    for train_index, test_index in kf.split(X):
        
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        
        select = SelectKBest(chi2,k=10)
        selected_features = select.fit(X_train, Y_train)
        indices_selected = selected_features.get_support(indices=True)
        colnames_selected = [X.columns[i] for i in indices_selected]

        X_train = X_train[colnames_selected]
        X_test = X_test[colnames_selected]

        
       
        model = SVC(kernel='rbf')
        model.fit(X_train, Y_train)
        y_hat = [x for x in model.predict(X_test)]
        auc.append(accuracy_score(Y_test, y_hat))

        cnf_matrix.append(confusion_matrix(Y_test, y_hat))  
     
    
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,train_size=0.8,random_state=15)
    select = SelectKBest(chi2,k=10)
    selected_features = select.fit(X_train, Y_train)
    indices_selected = selected_features.get_support(indices=True)
    colnames_selected = [X.columns[i] for i in indices_selected]

    X_train = X_train[colnames_selected]
    X_test = X_test[colnames_selected]
        
    model = SVC(kernel='rbf')
    model.fit(X_train, Y_train)
    y_hat = [x for x in model.predict(X_test)]
    auc.append(accuracy_score(Y_test, y_hat))
    cnf_matrix.append(confusion_matrix(Y_test, y_hat))  
    
    
    plt.figure()
    plot_confusion_matrix(cnf_matrix[0], classes=['Edible1','Poisonous1'],title='Confusion Matrix')
    plt.figure()
    plot_confusion_matrix(cnf_matrix[1], classes=['Edible2','Poisonous2'],title='Confusion Matrix')
    plt.figure()
    plot_confusion_matrix(cnf_matrix[2], classes=['Edible3','Poisonous3'],title='Confusion Matrix')
    plt.figure()
    plot_confusion_matrix(cnf_matrix[3], classes=['Edible4','Poisonous4'],title='Confusion Matrix')
    plt.figure()
    plot_confusion_matrix(cnf_matrix[4], classes=['Edible5','Poisonous5'],title='Confusion Matrix')
    plt.figure()
    plot_confusion_matrix(cnf_matrix[5], classes=['Edible6','Poisonous6'],title='Confusion Matrix')
    print('Acuraccy array: {auc}'.format(auc=auc))
    average = 0
    for x in auc:
        average += x
        
    average = average/6
    print('Accuracy average: {average}'.format(average=average))
    return auc

auc_svm_gaussian_KBest = svm_gaussian_KBest(X,Y)

"""

#SVM  PCA
"""
def svm_PCA(X, Y):
    
    pca = PCA(n_components=10)
    pca.fit(X)
    T = pca.transform(X)
    T= pd.DataFrame(T)                   

    print(T.head())
    
    print('\n')
    print('\n')

    kf = KFold(n_splits=5)
    auc = []
    cnf_matrix = []
  
    for train_index, test_index in kf.split(T):
        
        X_train, X_test = T.iloc[train_index], T.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

        
       
        model = SVC(kernel='linear')
        model.fit(X_train, Y_train)
        y_hat = [x for x in model.predict(X_test)]
        auc.append(accuracy_score(Y_test, y_hat))

        cnf_matrix.append(confusion_matrix(Y_test, y_hat))  
     
    
    X_train, X_test, Y_train, Y_test = train_test_split(T,Y,train_size=0.8,random_state=15)
        
    model = SVC(kernel='linear')
    model.fit(X_train, Y_train)
    y_hat = [x for x in model.predict(X_test)]
    auc.append(accuracy_score(Y_test, y_hat))
    cnf_matrix.append(confusion_matrix(Y_test, y_hat))  
    
    
    plt.figure()
    plot_confusion_matrix(cnf_matrix[0], classes=['Edible1','Poisonous1'],title='Confusion Matrix')
    plt.figure()
    plot_confusion_matrix(cnf_matrix[1], classes=['Edible2','Poisonous2'],title='Confusion Matrix')
    plt.figure()
    plot_confusion_matrix(cnf_matrix[2], classes=['Edible3','Poisonous3'],title='Confusion Matrix')
    plt.figure()
    plot_confusion_matrix(cnf_matrix[3], classes=['Edible4','Poisonous4'],title='Confusion Matrix')
    plt.figure()
    plot_confusion_matrix(cnf_matrix[4], classes=['Edible5','Poisonous5'],title='Confusion Matrix')
    plt.figure()
    plot_confusion_matrix(cnf_matrix[5], classes=['Edible6','Poisonous6'],title='Confusion Matrix')
    print('Acuraccy array: {auc}'.format(auc=auc))
    average = 0
    for x in auc:
        average += x
        
    average = average/6
    print('Accuracy average: {average}'.format(average=average))
    return auc

auc_svm_KBest = svm_PCA(X,Y)
"""


#SVM polynomial PCA
"""
def svm_polynomial_PCA(X, Y):
    
    pca = PCA(n_components=10)
    pca.fit(X)
    T = pca.transform(X)
    T= pd.DataFrame(T)                   

    print(T.head())
    
    print('\n')
    print('\n')

    kf = KFold(n_splits=5)
    auc = []
    cnf_matrix = []
  
    for train_index, test_index in kf.split(T):
        
        X_train, X_test = T.iloc[train_index], T.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

        
       
        model = SVC(kernel='poly', degree=4)
        model.fit(X_train, Y_train)
        y_hat = [x for x in model.predict(X_test)]
        auc.append(accuracy_score(Y_test, y_hat))

        cnf_matrix.append(confusion_matrix(Y_test, y_hat))  
     
    
    X_train, X_test, Y_train, Y_test = train_test_split(T,Y,train_size=0.8,random_state=15)
        
    model = SVC(kernel='poly', degree=4)
    model.fit(X_train, Y_train)
    y_hat = [x for x in model.predict(X_test)]
    auc.append(accuracy_score(Y_test, y_hat))
    cnf_matrix.append(confusion_matrix(Y_test, y_hat))  
    
    
    plt.figure()
    plot_confusion_matrix(cnf_matrix[0], classes=['Edible1','Poisonous1'],title='Confusion Matrix')
    plt.figure()
    plot_confusion_matrix(cnf_matrix[1], classes=['Edible2','Poisonous2'],title='Confusion Matrix')
    plt.figure()
    plot_confusion_matrix(cnf_matrix[2], classes=['Edible3','Poisonous3'],title='Confusion Matrix')
    plt.figure()
    plot_confusion_matrix(cnf_matrix[3], classes=['Edible4','Poisonous4'],title='Confusion Matrix')
    plt.figure()
    plot_confusion_matrix(cnf_matrix[4], classes=['Edible5','Poisonous5'],title='Confusion Matrix')
    plt.figure()
    plot_confusion_matrix(cnf_matrix[5], classes=['Edible6','Poisonous6'],title='Confusion Matrix')
    print('Acuraccy array: {auc}'.format(auc=auc))
    average = 0
    for x in auc:
        average += x
        
    average = average/6
    print('Accuracy average: {average}'.format(average=average))
    return auc

auc_svm_KBest = svm_polynomial_PCA(X,Y)
"""

#SVM kernel PCA
"""
def svm_kernel_PCA(X, Y):
    
    pca = PCA(n_components=10)
    pca.fit(X)
    T = pca.transform(X)
    T= pd.DataFrame(T)                   

    print(T.head())
    
    print('\n')
    print('\n')

    kf = KFold(n_splits=5)
    auc = []
    cnf_matrix = []
  
    for train_index, test_index in kf.split(T):
        
        X_train, X_test = T.iloc[train_index], T.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

        
       
        model = SVC(kernel='rbf')
        model.fit(X_train, Y_train)
        y_hat = [x for x in model.predict(X_test)]
        auc.append(accuracy_score(Y_test, y_hat))

        cnf_matrix.append(confusion_matrix(Y_test, y_hat))  
     
    
    X_train, X_test, Y_train, Y_test = train_test_split(T,Y,train_size=0.8,random_state=15)
        
    model = SVC(kernel='rbf')
    model.fit(X_train, Y_train)
    y_hat = [x for x in model.predict(X_test)]
    auc.append(accuracy_score(Y_test, y_hat))
    cnf_matrix.append(confusion_matrix(Y_test, y_hat))  
    
    
    plt.figure()
    plot_confusion_matrix(cnf_matrix[0], classes=['Edible1','Poisonous1'],title='Confusion Matrix')
    plt.figure()
    plot_confusion_matrix(cnf_matrix[1], classes=['Edible2','Poisonous2'],title='Confusion Matrix')
    plt.figure()
    plot_confusion_matrix(cnf_matrix[2], classes=['Edible3','Poisonous3'],title='Confusion Matrix')
    plt.figure()
    plot_confusion_matrix(cnf_matrix[3], classes=['Edible4','Poisonous4'],title='Confusion Matrix')
    plt.figure()
    plot_confusion_matrix(cnf_matrix[4], classes=['Edible5','Poisonous5'],title='Confusion Matrix')
    plt.figure()
    plot_confusion_matrix(cnf_matrix[5], classes=['Edible6','Poisonous6'],title='Confusion Matrix')
    print('Acuraccy array: {auc}'.format(auc=auc))
    average = 0
    for x in auc:
        average += x
        
    average = average/6
    print('Accuracy average: {average}'.format(average=average))
    return auc

auc_svm_KBest = svm_kernel_PCA(X,Y)
"""

# SVM ExtraTrees
"""
def svm_Trees(X, Y):
    
    
    print('\n')
    print('\n')

    kf = KFold(n_splits=5)
    auc = []
    cnf_matrix = []
  
    for train_index, test_index in kf.split(X):
        
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        
        

        select = ExtraTreesClassifier()
        select = select.fit(X_train, Y_train)
        trees = SelectFromModel(select, prefit=True)
        X_train = trees.transform(X_train)
        X_test = trees.transform(X_test)
       
        model = SVC(kernel='linear', class_weight={1:10})
        model.fit(X_train, Y_train)
        y_hat = [x for x in model.predict(X_test)]
        auc.append(accuracy_score(Y_test, y_hat))

        cnf_matrix.append(confusion_matrix(Y_test, y_hat))  
     
    
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,train_size=0.8,random_state=15)
    select = ExtraTreesClassifier()
    select = select.fit(X_train, Y_train)
    trees = SelectFromModel(select, prefit=True)
    X_train = trees.transform(X_train)
    X_test = trees.transform(X_test)
   
    
    
        
    model = SVC(kernel='linear', class_weight={1:10})
    model.fit(X_train, Y_train)
    y_hat = [x for x in model.predict(X_test)]
    auc.append(accuracy_score(Y_test, y_hat))
    cnf_matrix.append(confusion_matrix(Y_test, y_hat))  
    
    
    plt.figure()
    plot_confusion_matrix(cnf_matrix[0], classes=['Edible1','Poisonous1'],title='Confusion Matrix')
    plt.figure()
    plot_confusion_matrix(cnf_matrix[1], classes=['Edible2','Poisonous2'],title='Confusion Matrix')
    plt.figure()
    plot_confusion_matrix(cnf_matrix[2], classes=['Edible3','Poisonous3'],title='Confusion Matrix')
    plt.figure()
    plot_confusion_matrix(cnf_matrix[3], classes=['Edible4','Poisonous4'],title='Confusion Matrix')
    plt.figure()
    plot_confusion_matrix(cnf_matrix[4], classes=['Edible5','Poisonous5'],title='Confusion Matrix')
    plt.figure()
    plot_confusion_matrix(cnf_matrix[5], classes=['Edible6','Poisonous6'],title='Confusion Matrix')
    print('Acuraccy array: {auc}'.format(auc=auc))
    average = 0
    for x in auc:
        average += x
        
    average = average/6
    print('Accuracy average: {average}'.format(average=average))
    return auc

auc_lineal_regression_KBest = svm_Trees(X,Y)
"""
# SVM polynomial ExtraTrees
"""
def svm_polynomial_Trees(X, Y):
    
    
    print('\n')
    print('\n')

    kf = KFold(n_splits=5)
    auc = []
    cnf_matrix = []
  
    for train_index, test_index in kf.split(X):
        
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        
        

        select = ExtraTreesClassifier()
        select = select.fit(X_train, Y_train)
        trees = SelectFromModel(select, prefit=True)
        X_train = trees.transform(X_train)
        X_test = trees.transform(X_test)
       
        model = SVC(kernel='poly',degree=2)
        model.fit(X_train, Y_train)
        y_hat = [x for x in model.predict(X_test)]
        auc.append(accuracy_score(Y_test, y_hat))

        cnf_matrix.append(confusion_matrix(Y_test, y_hat))  
     
    
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,train_size=0.8,random_state=15)
    select = ExtraTreesClassifier()
    select = select.fit(X_train, Y_train)
    trees = SelectFromModel(select, prefit=True)
    X_train = trees.transform(X_train)
    X_test = trees.transform(X_test)
   
    
    
        
    model = SVC(kernel='poly', degree=2)
    model.fit(X_train, Y_train)
    y_hat = [x for x in model.predict(X_test)]
    auc.append(accuracy_score(Y_test, y_hat))
    cnf_matrix.append(confusion_matrix(Y_test, y_hat))  
    
    
    plt.figure()
    plot_confusion_matrix(cnf_matrix[0], classes=['Edible1','Poisonous1'],title='Confusion Matrix')
    plt.figure()
    plot_confusion_matrix(cnf_matrix[1], classes=['Edible2','Poisonous2'],title='Confusion Matrix')
    plt.figure()
    plot_confusion_matrix(cnf_matrix[2], classes=['Edible3','Poisonous3'],title='Confusion Matrix')
    plt.figure()
    plot_confusion_matrix(cnf_matrix[3], classes=['Edible4','Poisonous4'],title='Confusion Matrix')
    plt.figure()
    plot_confusion_matrix(cnf_matrix[4], classes=['Edible5','Poisonous5'],title='Confusion Matrix')
    plt.figure()
    plot_confusion_matrix(cnf_matrix[5], classes=['Edible6','Poisonous6'],title='Confusion Matrix')
    print('Acuraccy array: {auc}'.format(auc=auc))
    average = 0
    for x in auc:
        average += x
        
    average = average/6
    print('Accuracy average: {average}'.format(average=average))
    return auc

auc_lineal_regression_KBest = svm_polynomial_Trees(X,Y)
"""


# SVM kernel ExtraTrees
"""
def svm_kernel_Trees(X, Y):
    
    
    print('\n')
    print('\n')

    kf = KFold(n_splits=5)
    auc = []
    cnf_matrix = []
  
    for train_index, test_index in kf.split(X):
        
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        
        

        select = ExtraTreesClassifier()
        select = select.fit(X_train, Y_train)
        trees = SelectFromModel(select, prefit=True)
        X_train = trees.transform(X_train)
        X_test = trees.transform(X_test)
       
        model = SVC(kernel='rbf')
        model.fit(X_train, Y_train)
        y_hat = [x for x in model.predict(X_test)]
        auc.append(accuracy_score(Y_test, y_hat))

        cnf_matrix.append(confusion_matrix(Y_test, y_hat))  
     
    
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,train_size=0.8,random_state=15)
    select = ExtraTreesClassifier()
    select = select.fit(X_train, Y_train)
    trees = SelectFromModel(select, prefit=True)
    X_train = trees.transform(X_train)
    X_test = trees.transform(X_test)
   
    
    
        
    model = SVC(kernel='rbf')
    model.fit(X_train, Y_train)
    y_hat = [x for x in model.predict(X_test)]
    auc.append(accuracy_score(Y_test, y_hat))
    cnf_matrix.append(confusion_matrix(Y_test, y_hat))  
    
    
    plt.figure()
    plot_confusion_matrix(cnf_matrix[0], classes=['Edible1','Poisonous1'],title='Confusion Matrix')
    plt.figure()
    plot_confusion_matrix(cnf_matrix[1], classes=['Edible2','Poisonous2'],title='Confusion Matrix')
    plt.figure()
    plot_confusion_matrix(cnf_matrix[2], classes=['Edible3','Poisonous3'],title='Confusion Matrix')
    plt.figure()
    plot_confusion_matrix(cnf_matrix[3], classes=['Edible4','Poisonous4'],title='Confusion Matrix')
    plt.figure()
    plot_confusion_matrix(cnf_matrix[4], classes=['Edible5','Poisonous5'],title='Confusion Matrix')
    plt.figure()
    plot_confusion_matrix(cnf_matrix[5], classes=['Edible6','Poisonous6'],title='Confusion Matrix')
    print('Acuraccy array: {auc}'.format(auc=auc))
    average = 0
    for x in auc:
        average += x
        
    average = average/6
    print('Accuracy average: {average}'.format(average=average))
    return auc

auc_lineal_regression_KBest = svm_kernel_Trees(X,Y)
"""



# NN SeleckBest (number of hidden layers = media input and output)
"""
def NN_KBest(X, Y):
    
    
    print('\n')
    print('\n')

    kf = KFold(n_splits=5)
    auc = []
    cnf_matrix = []
  
    for train_index, test_index in kf.split(X):
        
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        
        

        select = SelectKBest(chi2,k=10)
        selected_features = select.fit(X_train, Y_train)
        indices_selected = selected_features.get_support(indices=True)
        colnames_selected = [X.columns[i] for i in indices_selected]

        X_train = X_train[colnames_selected]
        X_test = X_test[colnames_selected]
       
        model = MLPClassifier(hidden_layer_sizes=(10,10,10),max_iter=500)
        model.fit(X_train, Y_train)
        y_hat = [x for x in model.predict(X_test)]
        auc.append(accuracy_score(Y_test, y_hat))

        cnf_matrix.append(confusion_matrix(Y_test, y_hat))  
     
    
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,train_size=0.8,random_state=15)
    select = SelectKBest(chi2,k=10)
    selected_features = select.fit(X_train, Y_train)
    indices_selected = selected_features.get_support(indices=True)
    colnames_selected = [X.columns[i] for i in indices_selected]

    X_train = X_train[colnames_selected]
    X_test = X_test[colnames_selected]
    
    
        
    model = MLPClassifier(hidden_layer_sizes=(10,10,10),max_iter=500)
    model.fit(X_train, Y_train)
    y_hat = [x for x in model.predict(X_test)]
    auc.append(accuracy_score(Y_test, y_hat))
    cnf_matrix.append(confusion_matrix(Y_test, y_hat))  
    
    
    plt.figure()
    plot_confusion_matrix(cnf_matrix[0], classes=['Edible1','Poisonous1'],title='Confusion Matrix')
    plt.figure()
    plot_confusion_matrix(cnf_matrix[1], classes=['Edible2','Poisonous2'],title='Confusion Matrix')
    plt.figure()
    plot_confusion_matrix(cnf_matrix[2], classes=['Edible3','Poisonous3'],title='Confusion Matrix')
    plt.figure()
    plot_confusion_matrix(cnf_matrix[3], classes=['Edible4','Poisonous4'],title='Confusion Matrix')
    plt.figure()
    plot_confusion_matrix(cnf_matrix[4], classes=['Edible5','Poisonous5'],title='Confusion Matrix')
    plt.figure()
    plot_confusion_matrix(cnf_matrix[5], classes=['Edible6','Poisonous6'],title='Confusion Matrix')
    print('Acuraccy array: {auc}'.format(auc=auc))
    average = 0
    for x in auc:
        average += x
        
    average = average/6
    print('Accuracy average: {average}'.format(average=average))
    return auc

auc_lineal_regression_KBest = NN_KBest(X,Y)
"""
# NN SeleckBest (number of hidden layers = media input and output)
"""
def NN_PCA(X, Y):
    
    pca = PCA(n_components=10)
    pca.fit(X)
    T = pca.transform(X)
    T= pd.DataFrame(T)
    
    print('\n')
    print('\n')

    kf = KFold(n_splits=5)
    auc = []
    cnf_matrix = []
  
    for train_index, test_index in kf.split(T):
        
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        
       
        model = MLPClassifier(hidden_layer_sizes=(10,10,10),max_iter=500)
        model.fit(X_train, Y_train)
        y_hat = [x for x in model.predict(X_test)]
        auc.append(accuracy_score(Y_test, y_hat))

        cnf_matrix.append(confusion_matrix(Y_test, y_hat))  
     
    
    X_train, X_test, Y_train, Y_test = train_test_split(T,Y,train_size=0.8,random_state=15)
    
    
        
    model = MLPClassifier(hidden_layer_sizes=(10,10,10),max_iter=500)
    model.fit(X_train, Y_train)
    y_hat = [x for x in model.predict(X_test)]
    auc.append(accuracy_score(Y_test, y_hat))
    cnf_matrix.append(confusion_matrix(Y_test, y_hat))  
    
    
    plt.figure()
    plot_confusion_matrix(cnf_matrix[0], classes=['Edible1','Poisonous1'],title='Confusion Matrix')
    plt.figure()
    plot_confusion_matrix(cnf_matrix[1], classes=['Edible2','Poisonous2'],title='Confusion Matrix')
    plt.figure()
    plot_confusion_matrix(cnf_matrix[2], classes=['Edible3','Poisonous3'],title='Confusion Matrix')
    plt.figure()
    plot_confusion_matrix(cnf_matrix[3], classes=['Edible4','Poisonous4'],title='Confusion Matrix')
    plt.figure()
    plot_confusion_matrix(cnf_matrix[4], classes=['Edible5','Poisonous5'],title='Confusion Matrix')
    plt.figure()
    plot_confusion_matrix(cnf_matrix[5], classes=['Edible6','Poisonous6'],title='Confusion Matrix')
    print('Acuraccy array: {auc}'.format(auc=auc))
    average = 0
    for x in auc:
        average += x
        
    average = average/6
    print('Accuracy average: {average}'.format(average=average))
    return auc

auc_lineal_regression_KBest = NN_PCA(X,Y)
"""

# NN ExtraTrees

def NN_Trees(X, Y):
    
    
    print('\n')
    print('\n')

    kf = KFold(n_splits=5)
    auc_test = []
    auc_training = []
    cnf_matrix = []
  
    for train_index, test_index in kf.split(X):
        
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        
        

        select = ExtraTreesClassifier()
        select = select.fit(X_train, Y_train)
        trees = SelectFromModel(select, prefit=True)
        X_train = trees.transform(X_train)
        X_train =pd.DataFrame(X_train)
        print(X_train.head(5))
        X_test = trees.transform(X_test)
       
        model = MLPClassifier(hidden_layer_sizes=(10,10,10),max_iter=500)
        model.fit(X_train, Y_train)
        y_hat = [x for x in model.predict(X_test)]
        y_training = [x for x in model.predict(X_train)]
        auc_training.append(accuracy_score(Y_train, y_training))
        auc_test.append(accuracy_score(Y_test, y_hat))

        cnf_matrix.append(confusion_matrix(Y_test, y_hat))  
     
    
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,train_size=0.8,random_state=11)
    select = ExtraTreesClassifier()
    select = select.fit(X_train, Y_train)
    trees = SelectFromModel(select, prefit=True)
    X_train = trees.transform(X_train)
    X_test = trees.transform(X_test)
   
    
    
        
    model = MLPClassifier(hidden_layer_sizes=(20),max_iter=500)
    model.fit(X_train, Y_train)
    y_hat = [x for x in model.predict(X_test)]
    y_training = [x for x in model.predict(X_train)]
    auc_training.append(accuracy_score(Y_train, y_training))
    auc_test.append(accuracy_score(Y_test, y_hat))
    cnf_matrix.append(confusion_matrix(Y_test, y_hat))  
    
    
    plt.figure()
    plot_confusion_matrix(cnf_matrix[0], classes=['Edible1','Poisonous1'],title='Confusion Matrix')
    plt.figure()
    plot_confusion_matrix(cnf_matrix[1], classes=['Edible2','Poisonous2'],title='Confusion Matrix')
    plt.figure()
    plot_confusion_matrix(cnf_matrix[2], classes=['Edible3','Poisonous3'],title='Confusion Matrix')
    plt.figure()
    plot_confusion_matrix(cnf_matrix[3], classes=['Edible4','Poisonous4'],title='Confusion Matrix')
    plt.figure()
    plot_confusion_matrix(cnf_matrix[4], classes=['Edible5','Poisonous5'],title='Confusion Matrix')
    plt.figure()
    plot_confusion_matrix(cnf_matrix[5], classes=['Edible6','Poisonous6'],title='Confusion Matrix')
    print('Training accuracy array: {auc}'.format(auc=auc_training))
    print('Test acuraccy array: {auc}'.format(auc=auc_test))
    average = 0
    for x in auc_test:
        average += x
        
    average = average/6
    print('Accuracy average: {average}'.format(average=average))
    return auc_test

auc_lineal_regression_KBest = NN_Trees(X,Y)
"""
In general:

The number of hidden layer neurons are 2/3 (or 70% to 90%) of the size of the input layer.
The number of hidden layer neurons should be less than twice of the number of neurons in input layer.
The size of the hidden layer neurons is between the input layer size and the output layer size.
"""
