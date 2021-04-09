"""
Methods for sc-ML project 

Referenced to Pinkwink & sklearn-github

https://pinkwink.kr/
https://github.com/scikit-learn/scikit-learn

"""
# Authors : dokyum <github.com/dockyum>
#           EbraLim <github.com/EbraLim>
#           ryuseunghwan1 <github.com/ryuseunghwan1>

import time

from sklearn.model_selection import train_test_split

# model selection
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# pipeline
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

# scaler
from sklearn.preprocessing import RobustScaler

# models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC


# split_train_test
# clf_evaluation
# 


def split_train_test(df, 
                     test_size=0.2, 
                     random_state=42):
    """
    Split sc dataset into train and test subsets easily.
    
    
    Parameters
    ----------    
    df : DataFrame
    
    test_size : default 0.2
    
    random_state : default 42
    
    """
    train_set = df[df['test_set'] == 0]
    test_set = df[df['test_set'] == 1]

    X_train = train_set.drop(['test_set','fraud_YN'], axis=1)
    y_train = train_set['fraud_YN']

    X_test = test_set.drop(['test_set', 'fraud_YN'], axis=1)
    y_test = test_set['fraud_YN']
    
    print('split result')
    print('y_train : ', list(map(lambda x: x.tolist(),np.unique(y_train, return_counts=True))))
    print('y_test :', list(map(lambda x: x.tolist(),np.unique(y_test, return_counts=True))) )

    return X_train, X_test, y_train, y_test


def clf_evaluation(y_test, 
                   pred,
                   acc_s=True,
                   pre_s=True,
                   rec_s=True,
                   f1_s=True,
                   auc_s=True,
                   conf_m=True):
    """
    Get all evaluation scores.

    Parameters
    ----------
    y_test : 
        Truth labels
        
    pred : 
        Predicted labels
        
    acc_s, pre_s, rec_s, f1_s, auc_s, conf_m : bool, default=True
        On-off each scores
    
    """
    if acc_s :
        acc = accuracy_score(y_test, pred)
    else:
        acc = 0
    if pre_s :
        pre = precision_score(y_test, pred)
    else:
        pre = 0
    if rec_s :
        rec = recall_score(y_test, pred)
    else:
        rec = 0
    if f1_s :
        f1 = f1_score(y_test, pred)
    else:
        f1 = 0
    if auc_s :
        auc = roc_auc_score(y_test, pred)
    else:
        auc = 0
    if conf_m :
        confusion = confusion_matrix(y_test, pred)
    else:
        confusion = None
    
    print('=> confusion matrix')
    print(confusion)
    print('======================')
    
    print('Accurary: {0:.3f}, Precision: {1:.3f}'.format(acc, pre))
    print('Recall: {0:.3f}, F1: {1:.3f}, AUC:{2: .3f}'.format(rec, f1, auc))
    
    return acc, pre, rec, f1, auc
    
    
def get_result(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    
    return get_clf_eval(y_test, pred)


def get_result_pd(models, model_names, X_train, y_train, X_test, y_test):
    col_names = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    tmp = []
    
    for model in models:
        tmp.append(get_result(model[1], X_train, y_train, X_test, y_test))
        
    return pd.DataFrame(tmp, columns=col_names, index=model_names)


def draw_roc_curve(models, model_names, X_test, y_test):
    plt.figure(figsize=(10,10))
    
    for idx in range(len(models)):
        pred = models[idx][1].predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, pred)
        plt.plot(fpr, tpr, label=model_names[idx])
        
    plt.plot([0,1], [0,1], 'k--', label='random quess')
    plt.title('ROC')
    plt.legend()
    plt.grid()
    plt.show()

    
def fit_model(models, model_names, X_train, y_train, X_test, y_test):
    """
    models : models
    model_names : model_names
    X_train : X_train
    y_train : y_train
    X_test : X_test
    y_train : y_train
    """
    
    st_time = time.time()

    results = get_result_pd(models, model_names, X_train, y_train, X_test, y_test)

    print('Finish fitting!')
    print('Fit time :', time.time() - st_time)
    print('<Results>')
    print(results)
    draw_roc_curve(models, model_names, X_test, y_test)

"""
https://stackoverflow.com/questions/50285973/pipeline-multiple-classifiers?answertab=votes#tab-top
"""


class ClfSwitcher(BaseEstimator):
    """
    Classifier Switcher
    
    """
    def __init__(self, estimator = LogisticRegression(random_state=13)):
        """
        estimator : estimator
        """ 
        self.estimator = estimator


    def fit(self, X, y=None, **kwargs):
        self.estimator.fit(X, y)
        return self

 
    def predict(self, X, y=None):
        return self.estimator.predict(X)


    def predict_proba(self, X):
        return self.estimator.predict_proba(X)


    def score(self, X, y):
        return self.estimator.score(X, y)
    

# TODO : check estimator & each parameter value (like penalty) 아직 수정중입니다 완료하고 다시올리겠습니다ㅜ
def cross_validation():
    pipeline = Pipeline([
        ("scaler", RobustScaler()),
        ('clf', ClfSwitcher()),
    ])

    parameters = [
        {
            'clf__estimator': [LogisticRegression(random_state=13)], 
            'clf__estimator__penalty': ['l2', 'elasticnet', 'l1'],
        },
        {
            'clf__estimator': [DecisionTreeClassifier(random_state=13, max_depth=4)],
            'clf__estimator_max_depth' : [None,2,3,4],
        },
        {
            'clf__estimator': [RandomForestClassifier(random_state=13)],
            'clf__n_estimators': [100, 200, 500, 1000],
        },
        {
            'clf__estimator': [LinearSVC(random_state=13)],
            'clf__estimator__penalty': ['l2', 'elasticnet', 'l1'],
            'clf__estimator__loss': ['hinge', 'squared_hinge'],
        },
        {
            'clf__estimator' : [SVC(random_state=13)]
            'clf__estimator_kernel': ['poly',' rbf']
        }
    ]

    CV = GridSearchCV(pipeline, parameters, cv=5, n_jobs=12, return_train_score=False, verbose=3)
    return CV