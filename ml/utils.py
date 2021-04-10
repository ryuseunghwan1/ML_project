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

# TODO : 스케일러, 모델, 파라미터 점검
#
# Scaler
scaler = [('Robust', RobustScaler())]

# Estimator
clfs = [('LogisticReg', LogisticRegression(random_state=13)),
          ('DecisionTree', DecisionTreeClassifier(random_state=13)),
          ('RandomForest', RandomForestClassifier(random_state=13)),
          ('LightGBM', LGBMClassifier(random_state=13, boost_from_average=False)),
          ('SVC', SVC(random_state=13))]

clf_names = [clf[0] for clf in clfs]


# Pipelines
lr_pipe = Pipeline([("scaler", scaler[0][1]), ("clf", clfs[0][1])])
dt_pipe = Pipeline([("scaler", scaler[0][1]), ("clf", clfs[1][1])])
rf_pipe = Pipeline([("scaler", scaler[0][1]), ("clf", clfs[2][1])])
lgbm_pipe = Pipeline([("scaler", scaler[0][1]), ("clf", clfs[3][1])])
svm_pipe = Pipeline([("scaler", scaler[0][1]), ("clf", clfs[4][1])])
                   
                   
# parameters
lr_params = [{'clf__penalty': ['l2']}]                
dt_params = [{'clf__max_depth' : [None, 2, 3, 4]}]
rf_params = [{'clf__n_estimators': [1, 5, 10, 20], 'clf__max_depth' : [2, 3, 4, 5, 10, 50, 100]}]
lgbm_params = [{'clf__n_estimators' : [10, 30, 50, 100], 'clf__num_leaves': [4, 8, 16]}]
svc_params = [{'clf__kernel': ['poly','rbf']}]


# TODO : 완료했습니다. 검토 부탁드립니다.
def split_train_test(df, 
                     test_size=0.2, 
                     random_state=42):
    """
    Split sc dataset into train and test subsets easily.
    
    
    Parameters
    ----------    
    df : DataFrame
    
    test_size : default 0.2
    
    random_state : default 42 (random numder)
    
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


# TODO : 완료했습니다. 검토 부탁드립니다
def clf_evaluation(y_test, 
                   y_pred,
                   acc_s=True,
                   pre_s=True,
                   rec_s=True,
                   f1_s=True,
                   auc_s=True,
                   conf_m=True):
    """
    Get all evaluation scores from 'y_test', 'y_pred'.

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
        acc = accuracy_score(y_test, y_pred)
    else:
        acc = 0
    if pre_s :
        pre = precision_score(y_test, y_pred)
    else:
        pre = 0
    if rec_s :
        rec = recall_score(y_test, y_pred)
    else:
        rec = 0
    if f1_s :
        f1 = f1_score(y_test, y_pred)
    else:
        f1 = 0
    if auc_s :
        auc = roc_auc_score(y_test, y_pred)
    else:
        auc = 0
    if conf_m :
        confusion = confusion_matrix(y_test, y_pred)
        print('=> confusion matrix')
        print(confusion)
        print('======================')
    
    col_names = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    result = [[acc, pre, rec, f1, auc]]
    df_values = pd.DataFrame(result, columns=col_names)
    print(df_values)
    print('======================')
    
    return acc, pre, rec, f1, auc
    

# TODO : 
def dummy_selected(X_train, X_test, df):
    """
    df : 전처리를 마친 df를 넣어주세요
    X_train : 해당 df의 train data
    X_test : 해당 df의 test data
    """
    
    cat_features = ['car_model', 'sharing_type', 'age_group',
           'b2b', 'pf_type', 'start_hour','duration', 'accident_hour',
           'accident_location', 'acc_type1', 'insurance_site_aid_YN', 'police_site_aid_YN',
           'total_prsn_cnt']

    dum_features = [feature for feature in cat_features if feature in list(df.columns)]

    X_train_1hot = pd.get_dummies(X_train, columns=dum_features)
    X_test_1hot = pd.get_dummies(X_test, columns=dum_features)
    
    print('get dummies!')
    print('X_train : ', X_train_1hot.shape)
    print('X_test : ', X_test_1hot.shape)
    
    return X_train_1hot, X_test_1hot    


# gridsearchCV 완료했습니다 리뷰 부탁드립니다
def fit_cv(X_train, y_train, X_test, y_test, scoring='recall'):
    lr_CV = GridSearchCV(lr_pipe, lr_params, cv=5, scoring=scoring)
    dt_CV = GridSearchCV(dt_pipe, dt_params, cv=5, scoring=scoring)
    rf_CV = GridSearchCV(rf_pipe, rf_params, cv=5, scoring=scoring)
    lgbm_CV = GridSearchCV(lgbm_pipe, lgbm_params, cv=5, scoring=scoring)
    svc_CV = GridSearchCV(svm_pipe, svc_params, cv=5, scoring=scoring)

    CVs = [lr_CV, dt_CV, rf_CV, lgbm_CV, svc_CV]

    result_df = pd.DataFrame(columns=['classifier', 'train accuracy', 'train precision', 'train recall', 'train f1','train auc','test accuracy','test precision','test recall','test f1','test auc'])

    for idx, cv in enumerate(CVs):
        cv.fit(X_train, y_train)

        y_pred_train = cv.predict(X_train)
        y_pred_test = cv.predict(X_test)

        acc_tr, pre_tr, rec_tr, f1_tr, auc_tr = clf_evaluation(y_train, y_pred_train)
        acc_te, pre_te, rec_te, f1_te, auc_te = clf_evaluation(y_test, y_pred_test)

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
        
    result_df
    return result_df
    


# ClfSwitcher() : 포기

# Pipeline에 Classifier Switcher Class를 만들어 여러 모델을 한번에 학습시키려 시도
# 그 중 가장 좋은 n개의 estimator를 추출해서 testset으로 예측하여 각각의 acc, recall, precision 등을 파악하려 함
# 하지만 GridSearchCV는 1개의 모델(best_estimator)만 return한다

# 방법으로 cv_results_['rank_test_score'][:5] 를 통해 점수가 높은 5개의 estimator이름과 파라미터 값를 볼 수는 있으나,
# 모델을 만들려면 해당 파라미터 설정값을 다시 입력하여 다시 학습을 해야 하기에 시간이 오래 걸릴 것이라 판단
# 또한 점수 높은 모델이 LinearSVC만 중복하여 나오는 경우가 많음
# 따라서 각 모델마다 가장 좋은 파라미터 설정을 찾는 CV를 각각 만드는 방법을 선택
# 만약 비교 없이 1개의 모델만 활용 할 계획이면 사용 가능


class ClfSwitcher(BaseEstimator):
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

    # 코드 출처 : https://stackoverflow.com/questions/50285973/pipeline-multiple-classifiers?answertab=votes#tab-top


def fit_cross_validation(X_train, y_train, scoring='recall',*kwargs):
    """
    Fit X_train, y_train into estimator via GridSearchCV
    
    return
    ---------
        CV : fitted estimator

    Parameters
    ----------
    X_train
    
    y_train
    
    scoring : string, default='recall'
        'recall', 'precision', 'f1'
    
    """
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
            'clf__estimator': [DecisionTreeClassifier(random_state=13)],
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
            'clf__estimator' : [SVC(random_state=13)],
            'clf__estimator_kernel': ['poly',' rbf'],
        }
    ]

    CV = GridSearchCV(pipeline, parameters, cv=5, n_jobs=12, return_train_score=False, verbose=3, scoring=scoring)
    print_estimator = [print(i) for i in CV.best_estimator_.steps] 
    
    CV.fit(X_train, y_train)
    
    return CV


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