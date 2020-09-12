from imblearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, QuantileTransformer, PowerTransformer
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif, chi2
from sklearn.model_selection import GridSearchCV

from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import LocallyLinearEmbedding, TSNE, Isomap
from imblearn.over_sampling import SMOTE

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

from scipy import stats

def get_svc_grid(cv, dim_reduction_methods, scoring, random_state=None, n_jobs=1,
                 svc_kernel_l=None, svc_c_l=None, svc_gamma_l=None,
                 svc_class_weight_l=None):
    """
    Created GridSearchCV model with SVC estimator.

    Parameters
    -------
    cv : RepeatedStratifiedKFold

    dim_reduction_methods : list
        Methods of dimension reduction.
    scoring : str
        Scoring function for GridSearchCV.
    random_state : int, default=None
        Seed for randomness.
    n_jobs : int, default=1
        Number of cores should be used fot model training.
    svc_kernel_1 :

    svc_c_l :

    svc_gamma_l :

    svc_class_weight_l :

    """

    get_svc_grid.__doc__ = "A function returning pre-defined pipeline for svc binary classification"

    pipe = Pipeline([
        ("Fill_NaN", SimpleImputer(strategy="median")),
        ('StdScaler', StandardScaler()),
        ('oversampling', SMOTE(random_state = random_state,
                               n_jobs=n_jobs,
                               sampling_strategy='minority')),
        ('dim_reduction', SelectKBest(stats.ttest_ind)),
        ('classifier', SVC(probability=True, random_state=random_state)),])

    param_grid = {'dim_reduction': dim_reduction_methods,}
    if svc_kernel_l is not None:
        param_grid['classifier__kernel'] = svc_kernel_l
    if svc_class_weight_l is not None:
        param_grid['classifier__class_weight'] = svc_class_weight_l
    if svc_c_l is not None:
        param_grid['classifier__C'] = svc_c_l
    if svc_gamma_l is not None:
        param_grid['classifier__gamma'] = svc_gamma_l

    return GridSearchCV(
        estimator = pipe, param_grid = param_grid,
        scoring = scoring, cv = cv, n_jobs = n_jobs
    )


def get_lr_grid(cv, dim_reduction_methods, scoring, random_state=None, n_jobs=1,
                 lr_c_l=None, lr_penalty_l=None):
    """
    Created GridSearchCV model with LogisticRegression estimator.

    Parameters
    -------
    cv : RepeatedStratifiedKFold

    dim_reduction_methods : list
        Methods of dimension reduction.
    scoring : str
        Scoring function for GridSearchCV.
    random_state : int, default=None
        Seed for randomness.
    n_jobs : int, default=1
        Number of cores should be used fot model training.
    lr_c_l :

    lr_penalty_l :


    """

    pipe = Pipeline(
        [
            ("Fill_NaN", SimpleImputer(strategy="median")),
            ('StdScaler', StandardScaler()),
            ('oversampling', SMOTE(random_state = random_state,
                                   n_jobs=n_jobs,
                                   sampling_strategy='minority')),
            ('dim_reduction', SelectKBest(stats.ttest_ind)),
            ('classifier', LogisticRegression(solver='liblinear',
                                              random_state=random_state)),
        ]
    )

    param_grid = {'dim_reduction': dim_reduction_methods,}
    if lr_c_l is not None:
        param_grid['classifier__C'] = lr_c_l
    if lr_penalty_l is not None:
        param_grid['classifier__penalty'] = lr_penalty_l

    return GridSearchCV(
        estimator=pipe, param_grid=param_grid,
        scoring=scoring, cv=cv, n_jobs=n_jobs
    )


def get_rfc_grid(cv, dim_reduction_methods, scoring, random_state=None, n_jobs=1,
                 rfc_n_estimators_l=None, rfc_class_weight_l=None):
    """
    Created GridSearchCV model with RandomForestClassifier estimator.

    Parameters
    -------
    cv : RepeatedStratifiedKFold

    dim_reduction_methods : list
        Methods of dimension reduction.
    scoring : str
        Scoring function for GridSearchCV.
    random_state : int, default=None
        Seed for randomness.
    n_jobs : int, default=1
        Number of cores should be used fot model training.
    rfc_n_estimators_l :

    rfc_class_weight_l :

    """

    pipe = Pipeline([
        ("Fill_NaN", SimpleImputer(strategy="median")),
        ('StdScaler', StandardScaler()),
        ('oversampling', SMOTE(random_state = random_state,
                               n_jobs=n_jobs,
                               sampling_strategy='minority')),
        ('dim_reduction', SelectKBest(stats.ttest_ind)),
        ('classifier', RandomForestClassifier(random_state=random_state)),])

    param_grid = {'dim_reduction': dim_reduction_methods,}
    if rfc_n_estimators_l is not None:
        param_grid['classifier__n_estimators'] = rfc_n_estimators_l
    if rfc_class_weight_l is not None:
        param_grid['classifier__class_weight'] = rfc_class_weight_l

    return GridSearchCV(
        estimator = pipe, param_grid = param_grid,
        scoring = scoring, cv = cv, n_jobs = n_jobs
    )

def get_xgb_grid(cv, dim_reduction_methods, scoring, random_state=None, n_jobs=1,
                 xgb_kernel_l=None, xgb_gamma_l=None,
                 xgb_class_weight_l=None, xgb_lr_l=None):
    """
    Created GridSearchCV model with XGBClassifier estimator.

    Parameters
    -------
    cv : RepeatedStratifiedKFold

    dim_reduction_methods : list
        Methods of dimension reduction.
    scoring : str
        Scoring function for GridSearchCV.
    random_state : int, default=None
        Seed for randomness.
    n_jobs : int, default=1
        Number of cores should be used fot model training.
    xgb_kernel_l :

    xgb_gamma_l :

    xgb_class_weight_l :

    xgb_lr_l :

    """

    pipe = Pipeline([
        ("Fill_NaN", SimpleImputer(strategy="median")),
        ('StdScaler', StandardScaler()),
        ('oversampling', SMOTE(random_state = random_state,
                               n_jobs=n_jobs,
                               sampling_strategy='minority')),
        ('dim_reduction', SelectKBest(stats.ttest_ind)),
        ('classifier', XGBClassifier()),])

    param_grid = {'dim_reduction': dim_reduction_methods,}
    if xgb_lr_l is not None:
        param_grid['classifier__learning_rate'] = xgb_lr_l
    if xgb_kernel_l is not None:
        param_grid['classifier__kernel'] = xgb_kernel_l
    if xgb_gamma_l is not None:
        param_grid['classifier__gamma'] = xgb_gamma_l
    if xgb_class_weight_l is not None:
        param_grid['classifier__class_weight'] = xgb_class_weight_l

    return GridSearchCV(
        estimator = pipe, param_grid = param_grid,
        scoring = scoring, cv = cv, n_jobs = n_jobs
    )
