# Authors: retrieved from `sklearn`
#          Marina Pominova <marina.pominova@skoltech.ru>
#          Ekaterina Kondrateva <ekaterina.kondrateva@skoltech.ru>

import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.base import BaseEstimator, MetaEstimatorMixin, clone

from sklearn.utils.metaestimators import if_delegate_has_method

from sklearn.feature_selection._base import SelectorMixin 
from sklearn.feature_selection.from_model import _get_feature_importances

class SelectNFeaturesFromModel(BaseEstimator, SelectorMixin, MetaEstimatorMixin):
    """Feature selector that takes defined number of features for the dataset.
     OriThis feature selection algorithm looks only at the features (X), not the
    desired outputs (y), and can thus be used for unsupervised learning.
  
    Parameters
    ----------
    threshold : float, optional
        Features with a training-set variance lower than this threshold will
        be removed. The default is to keep all features with non-zero variance,
        i.e. remove the features that have the same value in all samples.
  
    Examples
    --------
        >>> X = [[0, 2, 0, 3], [0, 1, 4, 3], [0, 1, 1, 3]]
        >>> selector = SelectNFeaturesFromModel(model, n_features )
        >>> selector.fit_transform(X)
        array([[2, 0],
               [1, 4],
               [1, 1]])
    """

    def __init__(self, estimator, n_selected, prefit=False):
        self.estimator = estimator
        self.n_selected = n_selected
        self.prefit = prefit

    def _get_support_mask(self):
        if self.prefit:
            estimator = self.estimator
        elif hasattr(self, 'estimator_'):
            estimator = self.estimator_
        else:
            raise ValueError(
                'Either fit SelectFromModel before transform or set "prefit='
                'True" and pass a fitted estimator to the constructor.')
        scores = _get_feature_importances(estimator)
        threshold = np.sort(scores)[-self.n_selected]
        return scores >= threshold

    def fit(self, X, y=None, **fit_params):
        if self.prefit:
            raise NotFittedError(
                "Since 'prefit=True', call transform directly")
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y, **fit_params)
        return self
    
    @property
    def scores_(self):
        scores = _get_feature_importances(self.estimator_,)
        return scores

    @property
    def threshold_(self):
        scores = _get_feature_importances(self.estimator_,)
        return np.sort(scores)[-n_selected]
    
    @if_delegate_has_method('estimator')
    def partial_fit(self, X, y=None, **fit_params):
        if self.prefit:
            raise NotFittedError(
                "Since 'prefit=True', call transform directly")
        if not hasattr(self, "estimator_"):
            self.estimator_ = clone(self.estimator)
        self.estimator_.partial_fit(X, y, **fit_params)
        return self
