# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 13:03:08 2020

@author: Admin
"""

#%% importing library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn import metrics


from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import KFold,cross_val_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

#%% Data loading
data=pd.read_csv('SPAM text message 20170820 - Data.csv')
stemmer=PorterStemmer() # Stemming
lemmat=WordNetLemmatizer() # Lemmatization

x=data['Message'] # sms data
y=data['Category'] # target whether spam or ham
y= LabelEncoder().fit_transform(y) # Encoding target variable
mess=[]
for i in range(len(x)):
    review = re.sub('[^a-zA-Z]', ' ', x[i]) # Only words are considered and remaining other is replaced with space.
    review = review.lower() # Coverting sentence into lower case
    review = review.split() # Spliting sentence into words
    # porter stemming
    review = [stemmer.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review =' '.join(review) # Joining stemmed words to produce a sentence
    mess.append(review)

#%% Splitting dataset into train and test dataset
x_train,x_test,y_train,y_test=train_test_split(mess,y,test_size=0.2)
    
#%% converting text data into numbers using bag of words
bow=CountVectorizer(max_features=500)
bow.fit(x_train)
x_tr = bow.transform(x_train).toarray()
x_te = bow.transform(x_test).toarray()

scaler=StandardScaler()
scaler.fit(x_tr)
x_tr=scaler.transform(x_tr)
x_te=scaler.transform(x_te)

#%% TSNE plot for EDA 
def tsne_plot(x, y):
    tsne = TSNE(n_components=2, random_state=0)
    X_t = tsne.fit_transform(x)
    plt.figure(figsize=(12, 8))
    plt.scatter(X_t[np.where(y == 0), 0], X_t[np.where(y == 0), 1], marker='o', color='g', linewidth='1', alpha=0.8)
    plt.scatter(X_t[np.where(y == 1), 0], X_t[np.where(y == 1), 1], marker='o', color='r', linewidth='1', alpha=0.8) 
    plt.legend(loc='best')
    plt.legend(['ham','spam'])                           
    plt.title('TSNE plot',fontsize=20)

tsne_plot(x_tr,y_train)

def principal_component(x,y,n_components=2):
    n_pca=PCA(n_components=n_components).fit(x)
    pca_norm=n_pca.fit_transform(x)
    plt.figure()
    plt.scatter(pca_norm[np.where(y == 0), 0], pca_norm[np.where(y == 0), 1], marker='o', color='g', linewidth='1', alpha=0.8)
    plt.scatter(pca_norm[np.where(y == 1), 0], pca_norm[np.where(y == 1), 1], marker='o', color='r', linewidth='1', alpha=0.8) 
    plt.xlabel('1st Principal Component')
    plt.ylabel('2nd Principal Component')
    plt.title('PCA with first 2 features')
    plt.legend(['ham','spam'])                
    print('Total variance captured: ',sum(n_pca.explained_variance_ratio_))
    return n_pca,pca_norm

n_pca,red_pca=principal_component(x_tr,y_train,2) # pass dataframe 

#%% 
clf=SVC()
clf.fit(x_tr,y_train)
print('Training accuracy:',metrics.accuracy_score(y_train,clf.predict(x_tr)))
print('Test accuracy:',metrics.accuracy_score(y_test,clf.predict(x_te)))

#%% Using different classifier 
classifier=[GaussianNB,KNeighborsClassifier,AdaBoostClassifier,
            RandomForestClassifier,XGBClassifier]
def prediction(clf,x,y,k):
    pred=clf.predict_proba(x)
    pred_1=clf.predict(x)
    print('Accuracy Score',metrics.accuracy_score(y,pred_1))
    cv=KFold(n_splits=k,shuffle=True,random_state=33)
    scores = cross_val_score(clf, x, y, cv=cv)
    print("Average coefficient of determination using 5-foldcrossvalidation:",np.mean(scores))        
    yes_auc = roc_auc_score(y, pred[:,1])
    print('ROC AUC=%.3f' % (yes_auc))
    return pred_1

def best_classifer(x,y,k):
    pred=[]
    for i in range(len(classifier)):        
        print(classifier[i])
        clf=classifier[i]().fit(x,y)
        pred.append(prediction(clf,x,y,5))
    return pred

_=best_classifer(x_tr,y_train,5)

#%% Best classifer is XGBClassifier among other classifier, so carrying out hyperparameter tuning
param_dist = {'n_estimator':[100,1000,2000,3000],'learning_rate':[0.01,0.1,0.2,0.3]}
g_search = GridSearchCV(XGBClassifier(),param_grid = param_dist,n_jobs=-1)
g_search.fit(x_tr,y_train)
print(g_search.best_score_)
print(g_search.best_params_)

# Best paramater selected
clf=XGBClassifier(n_estimator=100,learning_rate=0.3).fit(x_tr,y_train)
train_pred=prediction(clf,x_tr,y_train,5)
test_pred=prediction(clf,x_te,y_test,5)


