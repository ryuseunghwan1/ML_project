"""
Methods for sc-ML project 

Referenced to Pinkwink & sklearn-github

https://pinkwink.kr/
https://github.com/scikit-learn/scikit-learn


Usecase
-------
import utils as ut


ut.split_train_test(socar_rd_cp)

"""
# Authors : dokyum <github.com/dockyum>
#           EbraLim <github.com/EbraLim>
#           ryuseunghwan1 <github.com/ryuseunghwan1>

#!pip install imblearn


import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# model selection
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.metrics import roc_curve

# sampler
from imblearn.over_sampling import SMOTE, SMOTENC, SMOTEN, ADASYN, BorderlineSMOTE, KMeansSMOTE, SVMSMOTE
from imblearn.over_sampling import RandomOverSampler

# pipeline
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

# scaler
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

# models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

# CV
from sklearn.model_selection import KFold, GridSearchCV, StratifiedKFold


# Sampler

samplers = [('SMOTE', SMOTE(random_state=13)),
            ('ADASYN', ADASYN(random_state=13)), 
            ('BorderlineSMOTE', BorderlineSMOTE(random_state=13)), 
            ('KMeansSMOTE', KMeansSMOTE(random_state=13)), 
            ('SVMSMOTE', SVMSMOTE(random_state=13)), 
            ('RandomOverSampler', RandomOverSampler(random_state=13))]

# Scaler
scalers = [('No', None ),
           ('RB', RobustScaler()),
           ('SD', StandardScaler()),
           ('MM', MinMaxScaler()),]

# Estimator
clfs = [('LogisticReg', LogisticRegression(random_state=13, max_iter=1000)),
          ('DecisionTree', DecisionTreeClassifier(random_state=13)),
          ('RandomForest', RandomForestClassifier(random_state=13)),
          ('LightGBM', LGBMClassifier(random_state=13)),
          ('SVC', SVC(random_state=13))]

clf_names = [clf[0] for clf in clfs]
                   
           
# parameters

lr_params = [{'clf__penalty': ['l2'], 
              'clf__class_weight' : [{0: 0.01, 1: 1.0}, {0: 0.005, 1: 1},'balanced']}]                
dt_params = [{'clf__max_depth' : [3, 4, 6, 8, 10, 30], 
              'clf__max_features': [None,'sqrt','log2'], 
              'clf__class_weight' : [{0: 0.01, 1: 1.0}, {0: 0.005, 1: 1},'balanced']}]
rf_params = [{'clf__n_estimators': [50, 100, 200, 400], 
              'clf__max_depth' : [4, 6, 8, 10, 30],
              'clf__class_weight' : [{0: 0.01, 1: 1.0}, {0: 0.005, 1: 1},'balanced']}]
lgbm_params = [{'clf__n_estimators' : [50, 100, 200, 400], 
                'clf__num_leaves': [4, 8, 16],
                'clf__class_weight' : [{0: 0.01, 1: 1.0}, {0: 0.005, 1: 1},'balanced']}]
svc_params = [{'clf__kernel': ['rbf'], 
               'clf__class_weight' : ['balanced'],
               'clf__C' : [0.1, 1.0]}]

          
# X_Train, X_Test, y_train, y_test 데이터 분리
def split_train_test(df):
    """
    Split sc dataset into train and test subsets easily.
    
    
    Parameters
    ----------    
    df : DataFrame
    
    
    return
    ----------    
    X_train, X_test, y_train, y_test
    
    .

    """
    train_set = df[df['test_set'] == 0]
    test_set = df[df['test_set'] == 1]

    X_train = train_set.drop(['c_25','c_1'], axis=1)
    y_train = train_set['fraud_YN']

    X_test = test_set.drop(['c_25', 'c_1'], axis=1)
    y_test = test_set['fraud_YN']
    
    print('==Split Result==')
    print('y_train : ', list(map(lambda x: x.tolist(), np.unique(y_train, return_counts=True))))
    print('y_test :', list(map(lambda x: x.tolist(), np.unique(y_test, return_counts=True))) )

    return X_train, X_test, y_train, y_test


# 선택한 Sampler 적용하여 데이터 샘플링
def fit_sampler(X_train, 
                y_train, 
                sampler='SMOTE'):
    """
    Sampler selector and fit_resample
    

    Parameters
    ----------
    X_train : 
        Train data of features
        
    y_train : 
        Train data of labels
        
    sampler : (string), default='SMOTE'
        'ADASYN', 'BorderlineSMOTE', 'KMeansSMOTE', 'SVMSMOTE', 'RandomOverSampler'

    Return
    ----------    
    X_train_over, y_train_over
    
    .
    
    """
    sampler_selected = [one[1] for one in samplers if one[0] == sampler][0]

    X_train_over, y_train_over = sampler_selected.fit_resample(X_train, y_train)

    print('==Sampling Result==')
    print('y_train : ', list(map(lambda x: x.tolist(), np.unique(y_train, return_counts=True))))
    print('y_train_over :', list(map(lambda x: x.tolist(), np.unique(y_train_over, return_counts=True))))

    return X_train_over, y_train_over


# Done
def clf_evaluation(y_test, 
                   y_pred,
                   acc_s=True,
                   pre_s=True,
                   rec_s=True,
                   f1_s=True,
                   auc_s=True,
                   conf_m=True,
                   view_scores=True):
    """
    Get all evaluation scores from 'y_test', 'y_pred'.

    Parameters
    ----------
    y_test : 
        Truth labels
        
    y_pred : 
        Predicted labels
        
    acc_s, pre_s, rec_s, f1_s, auc_s, conf_m : (bool), default=True
        On-off each scores
    
    """
    if acc_s :
        acc = accuracy_score(y_test, y_pred)
    else:
        acc = None
    if pre_s :
        pre = precision_score(y_test, y_pred)
    else:
        pre = None
    if rec_s :
        rec = recall_score(y_test, y_pred)
    else:
        rec = None
    if f1_s :
        f1 = f1_score(y_test, y_pred)
    else:
        f1 = None
    if auc_s :
        auc = roc_auc_score(y_test, y_pred)
    else:
        auc = None
    if conf_m :
        confusion = confusion_matrix(y_test, y_pred)
        print('=> confusion matrix')
        print(confusion)
        print('======================')
    
    if view_scores :
        col_names = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        result = [[acc, pre, rec, f1, auc]]
        df_values = pd.DataFrame(result, columns=col_names)
        print(df_values)
    print('====Done Evaluation====')
    
    return acc, pre, rec, f1, auc


# fit_cv
def fit_cv(X_train, y_train, X_test, y_test, scaler='RB', scoring='recall', conf_m=False, view_scores=False, draw_cv=True, n_jobs=-1, **kwargs):
    """
    GridSearchCV. 
    
    
    
    Parameters
    ----------

    scoring : (string), default='recall'
        'recall', 'precision', 'f1'
    
    scaler : (string), default='RB'
        'No': None, 'SD' : StandardScaler(), 'MM' : MinmaxScaler(), 'RB' : RobustScaler()
        
    conf_m : (bool), default=True
        on-off confusion matrix
        
    view_scores : (bool), default=True
        on-off each scores
    
    Return
    ----------
    
    cv_list : list of best estimators resulted from cross-validations.
    result_df : dataframe of cross-validation results.
    
    
    """
    cv_list = []
    cv_estimators = []
    st_time = time.time()
    
    scaler_selected = [one[1] for one in scalers if one[0] == scaler][0]
    
    # Pipelines
    lr_pipe = Pipeline([("scaler", scaler_selected), ("clf", clfs[0][1])], verbose=True)
    dt_pipe = Pipeline([("scaler", scaler_selected), ("clf", clfs[1][1])], verbose=True)
    rf_pipe = Pipeline([("scaler", scaler_selected), ("clf", clfs[2][1])], verbose=True)
    lgbm_pipe = Pipeline([("scaler", scaler_selected), ("clf", clfs[3][1])], verbose=True)
    svm_pipe = Pipeline([("scaler", scaler_selected), ("clf", clfs[4][1])], verbose=True)
    
    skfold = StratifiedKFold(n_splits=5, random_state=13, shuffle=True)
    
    lr_CV = GridSearchCV(lr_pipe, lr_params, cv=skfold, scoring=scoring, n_jobs=n_jobs)
    dt_CV = GridSearchCV(dt_pipe, dt_params, cv=skfold, scoring=scoring, n_jobs=n_jobs)
    rf_CV = GridSearchCV(rf_pipe, rf_params, cv=skfold, scoring=scoring, n_jobs=n_jobs)
    lgbm_CV = GridSearchCV(lgbm_pipe, lgbm_params, cv=skfold, scoring=scoring, n_jobs=n_jobs)
    svc_CV = GridSearchCV(svm_pipe, svc_params, cv=skfold, scoring=scoring, n_jobs=n_jobs)

    CVs = [lr_CV, dt_CV, rf_CV, lgbm_CV, svc_CV]

    result_df = pd.DataFrame(columns=['classifier', 'train accuracy', 'train precision', 'train recall', 'train f1','train auc','test accuracy','test precision','test recall','test f1','test auc'])

    for idx, cv in enumerate(CVs):
        cv.fit(X_train, y_train)
        
        cv_list.append(cv.best_estimator_)
        cv_estimators.append([cv.best_estimator_[0], cv.best_estimator_[1]])

        y_pred_train = cv.predict(X_train)
        y_pred_test = cv.predict(X_test)

        acc_tr, pre_tr, rec_tr, f1_tr, auc_tr = clf_evaluation(y_train, y_pred_train, conf_m=conf_m, view_scores=view_scores, **kwargs)
        acc_te, pre_te, rec_te, f1_te, auc_te = clf_evaluation(y_test, y_pred_test, conf_m=conf_m, view_scores=view_scores, **kwargs)

        result = {'classifier' : clfs[idx][0],
                  'train accuracy' : acc_tr,
                  'train precision' : pre_tr,
                  'train recall' : rec_tr,
                  'train f1': f1_tr,
                  'train auc' : auc_tr,
                  'test accuracy' : acc_te,
                  'test precision' : pre_te,
                  'test recall' : rec_te,
                  'test f1' : f1_te,
                  'test auc' : auc_te }

        result_df = result_df.append(result, ignore_index=True)
        
        
        # 히트맵
        conf_mtx = confusion_matrix(y_test, y_pred_test)
        plt.figure(figsize=(6,4))
        plt.title(f"< {clfs[idx][0]} >")
        sns.heatmap(conf_mtx, annot=True, yticklabels=["No_act", "Yes_act"], xticklabels=["No_pred", "Yes_pred"], fmt='d')
        plt.show()
        
    print('Fit time :', round((time.time() - st_time) / 60, 2), 'min')
    
    if draw_cv:
        draw_roc_curve(cv_list, cv_estimators, X_test, y_test)
        
    result_df
    return cv_list, result_df
    

# roc_curve 그래프 만들기
def draw_roc_curve(models, model_names, X_test, y_test):
    plt.figure(figsize=(10,10))
    
    for idx in range(len(models)-1):
        pred = models[idx].predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, pred)
        plt.plot(fpr, tpr, label=model_names[idx])
        
    plt.plot([0,1], [0,1], 'k--', label='random quess')
    plt.title('ROC')
    plt.legend()
    plt.grid()
    plt.show()