from .select_n_features import SelectNFeaturesFromModel
from .classification_grid import get_svc_grid, get_lr_grid, get_rfc_grid, get_xgb_grid

# sklearn
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, StratifiedShuffleSplit, cross_val_score, cross_val_predict, GridSearchCV, LeaveOneOut
from sklearn.feature_selection import SelectFromModel, VarianceThreshold, SelectKBest, f_classif, chi2

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, QuantileTransformer, PowerTransformer
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import LocallyLinearEmbedding, TSNE, Isomap


### metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, confusion_matrix
try:
    from sklearn.externals import joblib
except:
    import joblib

# plotting
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
sns.set_style('darkgrid')

# others
import pandas as pd
import numpy as np
import time
from datetime import date
from scipy import stats
from mlxtend.evaluate import bootstrap_point632_score
from collections import Counter, defaultdict
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import logging


logging.basicConfig(filename= 'D:\YandexDisk\kondratevakate\01_skoltech\Brain\sbi-platform-2019\sbi-classification-pipeline\neurobot\neurobot_logging.log', level = logging.DEBUG)
logger = logging.getLogger()
logger.info('The LINEAR started.')


def print_results_(clf_grid_dict, save_plot_to=None, problem_name=None):
    """
    Shows result of algorithms.

    Parameters
    -------
    clf_grid_dict :
        Dictionary of classifiers.
    save_plot_to : str, default=None
        The path to the plot file if it should be saved.
    problem_name : str, optional
        Classificator name for saving model.
    """

    results = {
            "classifier" : [],
            "best parameters" : [],
            "best dim. reduction method" : [],
            "mean" : [],
            "std" : []
           }

    for clf, grid in clf_grid_dict.items():
        results["classifier"].append(clf)
        results["best parameters"].append(
            ", ".join(
                [param + " = " + str(best_value) for param, best_value in grid.best_params_.items() if param != 'dim_reduction']
            )
        )
        results["best dim. reduction method"].append(grid.best_params_['dim_reduction'])
        idx = grid.best_index_
        results["mean"].append(grid.cv_results_['mean_test_score'][idx])
        results["std"].append(grid.cv_results_['std_test_score'][idx])

    results = pd.DataFrame(
        results, columns=["classifier", "best parameters", "best dim. reduction method", "mean", "std"]
    )

    plt.figure(figsize=[8,len(clf_grid_dict)*3])
    display(results.set_index("classifier"))

    today = date.today().strftime("%d%m%Y")

    # draw graph
    fig = plt.figure()
    for i in results.index:
        plt.bar(i, results.loc[i, "mean"], yerr=results.loc[i, "std"], label=results.loc[i, "classifier"])

    plt.xticks(range(results.shape[0]), results.loc[:, "classifier"])
    plt.axis(ymin=0.0, ymax=1.0)
    if save_plot_to is not None:
        fig.savefig('{}{}_{}_clf.png'.format(save_plot_to,
                                              today,
                                              problem_name
                                             ), bbox_inches='tight')

    cnt = 0
    fig, axs = plt.subplots(len(results["classifier"]), figsize=(8,len(clf_grid_dict)*4))
    for clf in results["classifier"]:
        graph_mean = []
        graph_std = []
        j = 0
        for n in range(25):
            graph_mean.append(clf_grid_dict[clf].cv_results_['split{}_test_score'.format(n)].mean())
            graph_std.append(clf_grid_dict[clf].cv_results_['split{}_test_score'.format(n)].std())
        try:
            ax = axs[cnt]
        except TypeError:
            ax = axs
        for i in range(5):
            ax.bar(range(j, j+5), [graph_mean[m] for m in range(j, j+5)], yerr=[graph_std[m] for m in range(j, j+5)])
            j += 5
        ax.set_xticks(np.linspace(2,22,5))
        ax.set_xticklabels(range(1,6))
        ax.set_title(results["classifier"][cnt].upper())
        try:
            for ax in axs.flat:
                ax.set(xlabel='5 folds with 5 iterations for each', ylabel='Mean score for each iteration')
        except AttributeError:
            ax.set(xlabel='5 folds with 5 iterations for each', ylabel='Mean score for each iteration')
        cnt += 1
    if save_plot_to is not None:
        fig.savefig('{}{}_{}_folds.png'.format(save_plot_to, today, problem_name), bbox_inches='tight')
    plt.show()

    print("Best model: ")
    clf = results.loc[results["mean"].argmax(), "classifier"]
    print(clf)
    print("\n".join([param + " = " + str(best_value) for param, best_value in clf_grid_dict[clf].best_params_.items()]))


class GridCVLinear:
    """
    A class used to search among several classifiers with different assesement.
    Also provides instruments for dimension reduction and drawing graphs.

    ...

    Attributes
    ----------
    X : pandas.DataFrame
        The training data.
    y : pandas.DataFrame
        The target to training data.
    problem_name : str, optional
        Classificator name for saving model and meta-files.
    classifiers : list, default=['lr', 'rfc', 'svc', 'xgb']
        The list of classifiers should be used.
    n_splits : int, default=5

    n_repeats : int, default=5

    n_bootstrap_splits : int, default=100
        Number of bootstrap splits.
    scoring : str, default='roc_auc'
        The scoring should be used while training models.
    random_state : int, default=42
        The parameter which defines the randomness of the algorithm.
    n_jobs : int, default=-1
        Maximum number of cores used by the algorithm.
    non_l_dim_r: bool, default=False


    Methods
    -------
    train()
        Performs the grid search among classifiers
    save_best_models(path='')
        Saves best models to dedicated path
    print_results()
        Displays the best models with hyperparameters chosen
    train():
        Performs the grid search among classifiers
    print_results_(save_plot_to=None)
        Provides results of algorithm and saves if it is necessary.
    print_results(model=None)
        Plots results.
    loo_cv():
        Performs Leave-One-Out cross validation.
    bootstrap_632(n_splits=self.n_bootstrap_splits)
        Performs bootstrap validation. 1000 iteration are good for small datasets,
        for large datasets we can skip bootstraping.
    plot_val(save_fig=True, fig_name='val')
        Plots validation results and saves the figure.
    save_val_results(problem_name = '', path='')
        Saves validation results.
    save_models_pkl( problem_name='', path='')
        Saves models in pkl format.
    train_val(save=True, val=True, plot=True, save_fig=True,
               problem_name='', path='', fig_name='val')
        Perfomes training, validation and results printing.
        Saves models and validation results.

    """

    def __init__(self, X, y,
                 problem_name = 'test_classification',
                 classifiers = ['lr', 'rfc', 'svc', 'xgb'],
                 n_splits = 5,
                 n_repeats = 5,
                 n_bootstrap_splits = 100,
                 scoring = 'roc_auc',
                 random_state = 42,
                 n_jobs = -1,
                 # xgb = False,
                 non_l_dim_r = False
                ):
        if not classifiers:
            raise ValueError("Calassifiers list should not be empty")

        self.X = X
        self.y = y
        logger.info('Input files reading...')
        self.problem_name = problem_name
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.n_bootstrap_splits = n_bootstrap_splits
        self.scoring = scoring
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.pos_label = None
        self.features_groups = []
        self.save_plot_to = None
        self.path = ''
        self.n_objects = self.X.shape[0]
        # self.classifiers = ['best', 'svc', "lr", "rfc"]
        self.classifiers = classifiers
        # self.xgb = xgb
        self.non_l_dim_r = non_l_dim_r
        self.grid = {}
        self.loo_results = []
        self.bootstrap_results = []

    def train(self):
        """ Performs the grid search among classifiers
        """

        print ('Number of samples ', self.X.shape[0], "\n")
        print ('Number of features ', self.X.shape[1], "\n")

        if isinstance(self.X, np.ndarray):
            self.X = pd.DataFrame(self.X)
            self.y = pd.DataFrame(self.y)[0]

        cv = RepeatedStratifiedKFold(
            n_splits = self.n_splits,
            n_repeats = self.n_repeats,
            random_state = self.random_state,
        )

        if self.X.shape[1] > self.X.shape[0]:
            n_features = [int(self.X.shape[0]*0.8), self.X.shape[0]]

            dim_reduction_methods = []
            dim_reduction_methods += [SelectKBest(f_classif, n) for n in n_features]
            dim_reduction_methods += [SelectNFeaturesFromModel(
                RandomForestClassifier(
                    n_estimators = int(self.X.shape[0]**0.5),
                    random_state = self.random_state), n
            ) for n in n_features]

            dim_reduction_methods += [SelectNFeaturesFromModel(
                LogisticRegression(
                    random_state = self.random_state), n
            ) for n in n_features]

            dim_reduction_methods += [PCA(0.95, random_state = self.random_state)]

            if self.non_l_dim_r:
                n_components = [2, self.X.shape[0]*0.1]
                dim_reduction_methods += [Isomap(n_n, n_c, n_jobs=self.n_jobs,
                                                ) for n_n in n_neighbors for n_c in n_components]
        else:
            dim_reduction_methods = []
            dim_reduction_methods += [SelectKBest(f_classif, k='all')]


        print("Target distribution: ")
        print(self.y.value_counts(), "\n")

        if self.pos_label is None:
            y_enc = pd.Series(
                LabelEncoder().fit_transform(self.y), index = self.y.index
            )
        else:
            y_enc = pd.Series(
                self.y == pos_label, dtype=int
            )

        features_weight = self.y.value_counts().to_dict()
        max_feature = max(list(features_weight.values()))
        weights = {k: max_feature // v for k, v in features_weight.items()}

        # Perform the search for all selected models
        if 'svc' in self.classifiers:
            print("Training SVC(linear)...")
            grid_cv_svc = get_svc_grid(
                cv, dim_reduction_methods, self.scoring,
                random_state = self.random_state,
                n_jobs = self.n_jobs,
                svc_kernel_l = ["linear"],
                svc_class_weight_l = [weights],
                svc_c_l = [10 ** i for i in range(1, 4, 1)],
                svc_gamma_l = [10 ** i for i in range(-3, -1, 1)]
            )
            start_time = time.time()
            grid_cv_svc.fit(self.X, y_enc)
            self.grid['SVC'] = grid_cv_svc
            print("(training took {}s)\n".format(time.time() - start_time))


        if 'lr' in self.classifiers:
            print("Training LR...")
            grid_cv_lr = get_lr_grid(
                cv, dim_reduction_methods, self.scoring,
                random_state = self.random_state,
                n_jobs = self.n_jobs,
                lr_c_l = [10 ** i for i in range(-4, -1, 1)],
                lr_penalty_l = ["l1", "l2"]
            )
            start_time = time.time()
            grid_cv_lr.fit(self.X, y_enc)
            self.grid['LR'] = grid_cv_lr
            print("(training took {}s)\n".format(time.time() - start_time))


        if 'rfc' in self.classifiers:
            print("Training RFC...")
            step = (self.X.shape[0] - int(self.X.shape[0]**0.5))//4
            grid_cv_rfc = get_rfc_grid(
                cv, dim_reduction_methods, self.scoring,
                random_state = self.random_state, n_jobs=self.n_jobs,
                rfc_class_weight_l = [weights],
                rfc_n_estimators_l = [i for i in range(int(self.X.shape[0]**0.5),
                                                       self.X.shape[0], step)]
            )
            start_time = time.time()
            grid_cv_rfc.fit(self.X, y_enc)
            self.grid['RFC'] = grid_cv_rfc
            print("(training took {}s)\n".format(time.time() - start_time))

        if 'xgb' in self.classifiers:
            # self.classifiers.append('xgboost')
            print("Training XGBoost(linear)...")
            grid_cv_xgb = get_xgb_grid(
                cv, dim_reduction_methods, self.scoring,
                random_state = self.random_state,
                n_jobs = self.n_jobs,
                xgb_kernel_l = ["gbtree"],
                xgb_gamma_l = [10 ** i for i in range(-3, -1, 1)],
                xgb_class_weight_l = [int(np.round(self.y.value_counts()[0] / self.y.value_counts()[1]))]
            )
            start_time = time.time()
            grid_cv_xgb.fit(self.X, y_enc)
            self.grid['XGBoost'] = grid_cv_xgb
            print("(training took {}s)\n".format(time.time() - start_time))

            # best_model = max(
            #     [grid_cv_svc, grid_cv_lr, grid_cv_rfc, grid_cv_xgb], key=lambda x: x.best_score_
            # ).best_estimator_
            # self.grid = [best_model, grid_cv_svc, grid_cv_lr, grid_cv_rfc, grid_cv_xgb]
            # return self.grid

        # else:
        #     best_model = max(
        #         [grid_cv_svc, grid_cv_lr, grid_cv_rfc], key=lambda x: x.best_score_
        #     ).best_estimator_
        #     self.grid = [best_model, grid_cv_svc, grid_cv_lr, grid_cv_rfc]
        #     return self.grid

        # Finds the best model to show it in the results
        best_model = max(self.grid.values(), key=lambda x: x.best_score_).best_estimator_
        return list(self.grid.values()) + [best_model]

    def print_results_(self, save_plot_to=None):
        """
        Provides results of algorithm and saves if it is necessary.

        Parameters
        -------
        save_plot_to : str, default=None
            Path to the target file.
        """

        clf_grid_dict = dict(zip(self.classifiers, self.grid))

        results = {
                "classifier" : [],
                "best parameters" : [],
                "best dim. reduction method" : [],
                "mean" : [],
                "std" : []
               }

        for clf, grid in clf_grid_dict.items():
            results["classifier"].append(clf)
            results["best parameters"].append(
                ", ".join(
                    [param + " = " + str(best_value) for param, best_value in grid.best_params_.items() if param != 'dim_reduction']
                )
            )
            results["best dim. reduction method"].append(grid.best_params_['dim_reduction'])
            idx = grid.best_index_
            results["mean"].append(grid.cv_results_['mean_test_score'][idx])
            results["std"].append(grid.cv_results_['std_test_score'][idx])

        results = pd.DataFrame(
            results, columns=["classifier", "best parameters", "best dim. reduction method", "mean", "std"]
        )
        display(results.set_index("classifier"))


    def print_results(self, model=None):
        """
        Plots results.
        """
        print_results_(self.grid)
        # if self.xgb:
        #     print_results_(
        #         {
        #             'SVC' : self.grid[1],
        #             "LR" : self.grid[2],
        #             "RFC" : self.grid[3],
        #             "XGBoost" : self.grid[4]
        #         }
        #     )
        # else:
        #     print_results_(
        #         {
        #             'SVC' : self.grid[1],
        #             "LR" : self.grid[2],
        #             "RFC" : self.grid[3]
        #         }
        #     )

    def loo_cv(self):
        """
        Performs Leave-One-Out cross validation.
        """

        print (self.problem_name)
        self.loo_results = []
        # for k in range(1, len(self.grid)):
        for key in self.grid.keys():
            start_time = time.time()
            # best_model = self.grid[k].best_estimator_
            best_model = self.grid[key].best_estimator_
            loo = LeaveOneOut()
            loo.get_n_splits(self.X)
            predict = []

            for train_index, test_index in loo.split(self.X):
                X_train, X_test = self.X.loc[train_index], self.X.loc[test_index]
                y_train, y_test = self.y.loc[train_index], self.y.loc[test_index]
                predict.append(best_model.fit(X_train, y_train).predict(X_test)[0])

            tpr, fpr, fnr, tnr = (confusion_matrix(self.y,
                                                   predict
                                                   ).astype('float')
                                  /confusion_matrix(self.y,
                                                    predict
                                                    ).sum(axis=1)[:, np.newaxis]
                                 ).ravel()
            end_time = np.round(time.time() - start_time, 2)
            self.loo_results.append([np.round(((tpr + tnr)*100) / 2, 2),
                                     np.round(tpr*100, 2), np.round(tnr*100, 2),
                                     end_time])
            print(
                # self.classifiers[k].upper()  + ': ',
                key + ': ',
                ' acc', np.round((tpr + tnr) / 2, 2),
                ' tpr', np.round(tpr, 2),
                ' tnr', np.round(tnr, 2),
                ' time', end_time
            )

    def bootstrap_632(self, n_splits=None):
        """
        Performs bootstrap validation. 1000 iteration are good for small datasets,
        for large datasets we can skip bootstraping.
        """
        if n_splits == None:
            n_splits = self.n_bootstrap_splits

        print (self.problem_name)
        self.bootstrap_results = []
        # for k in range(1, len(self.grid)):
        for key in self.grid.keys():
            start_time = time.time()

            if isinstance(self.X, np.ndarray):
                scores = bootstrap_point632_score(
                    # self.grid[k].best_estimator_, self.X,
                    self.grid[key].best_estimator_, self.X,
                    self.y, n_splits=n_splits,
                    method='.632', random_seed=42
                )
            else:
                scores = bootstrap_point632_score(
                # self.grid[k].best_estimator_, self.X.values,
                self.grid[key].best_estimator_, self.X.values,
                self.y.values, n_splits=n_splits,
                method='.632', random_seed=42
            )


            acc = np.mean(scores)
            lower = np.percentile(scores, 2.5)
            upper = np.percentile(scores, 97.5)
            end_time = np.round(time.time() - start_time, 2)
            self.bootstrap_results.append([np.round(100*acc, 2),
                                           [np.round(100*lower, 2),
                                            np.round(100*upper, 2)],
                                           end_time])
            print(
                # self.classifiers[k].upper(),
                key,
                ' acc: %.2f%%' % (100*acc),
                ' 95%% Confidence interval: [%.2f, %.2f]' % \
                    (100*lower, 100*upper),
                ' time', end_time
            )


    def plot_val(self, save_fig=True, fig_name='val'):
        """
        Plots validation results and saves the figure.

        Parameters
        ----------
        fig_name : str
            Name of the figure to be saved.
        """

        min_y = min([self.loo_results[i][1] for i in range(len(self.loo_results))])
        fig = plt.figure(figsize=(12,6))
        for i in range(len(self.bootstrap_results)):
            a, = plt.plot(i, self.bootstrap_results[i][0], "o",
             color="pink", label='bootstrap acc')
            b = plt.vlines(i, self.bootstrap_results[i][1][0],
                              self.bootstrap_results[i][1][1],
                              colors='grey', label='conf. interval')
        first_legend = plt.legend(handles=[a, b],
                                  loc='lower left')
        plt.gca().add_artist(first_legend)

        for i in range(len(self.loo_results)):
            c, = plt.plot(i+0.25, self.loo_results[i][0], "*",
                  color="darkred", label='loo acc')
            d, = plt.plot(i+0.25, self.loo_results[i][1], "x",
                  color="black", label='tnr')
            e, = plt.plot(i+0.25, self.loo_results[i][2], "+",
                 color="g", label='tpr')
        second_legend = plt.legend(handles=[c, d, e], loc='lower right')
        plt.gca().add_artist(second_legend)

        for i in range(len(self.classifiers[1:])-1):
            plt.vlines(i+0.65, min_y-15, 100, linestyles='dashdot', colors='white', alpha=0.5)

        plt.xticks(np.linspace(0.125, len(self.classifiers[1:])-0.875,
                   len(self.classifiers[1:])),
                   [i.upper() for i in self.classifiers[1:]])
        plt.ylabel('Percentage')
        plt.xlabel('Classifier')
        plt.title('LeaveOneOut and Bootstrap validation results')
        plt.grid(axis='x')
        plt.show()
        if save_fig == True:
            fig.savefig(fig_name + '.png')


    def save_val_results(self, problem_name = '', path=''):
        """Saves validation results.

        Parameters
        ----------
        problem_name : str
            Dataset name and problem type.
        path : str
            Path to the folder where models should be saved.
        """

        tree = lambda: defaultdict(tree)
        model_param = tree()
        for i, clf in enumerate(self.classifiers[1:]):
            model_param['LeaveOneOut'][clf.upper()]['acc'] = self.loo_results[i][0]
            model_param['LeaveOneOut'][clf.upper()]['tpr'] = self.loo_results[i][1]
            model_param['LeaveOneOut'][clf.upper()]['tnr'] = self.loo_results[i][2]
            model_param['LeaveOneOut'][clf.upper()]['time'] = self.loo_results[i][3]
            model_param['Bootstrap'][clf.upper()]['acc'] = self.bootstrap_results[i][0]
            model_param['Bootstrap'][clf.upper()]['Confidence interval'] = self.bootstrap_results[i][1]
            model_param['Bootstrap'][clf.upper()]['time'] = self.bootstrap_results[i][2]
        json_file = json.dumps(model_param, indent=4)
        today = date.today().strftime("%d%m%Y")
        with open('{}_{}_{}_val_results.json'.format(today,
                                                     problem_name,
                                                     self.n_objects),
                  'w') as file:
            file.write(json_file)


    def save_models_pkl(self, problem_name='', path=''):
        """Saves models in pkl format.

        Parameters
        ----------
        problem_name : str
            Dataset name and problem type
        path : str
            Path to the folder where models should be saved.
        """

        today = date.today().strftime("%d%m%Y")
        for i, clf in enumerate(self.classifiers[1:]):
            model = self.grid[i+1]
            idx = model.best_index_
            mean = model.cv_results_['mean_test_score'][idx]
            std = model.cv_results_['std_test_score'][idx]
            with open(path + "{}_{}_{}_{}_{}_{}.pkl".format(today, problem_name,
                                                  self.n_objects,
                                                  clf.upper(), int(mean*100),
                                                  int(std*100)), "wb") as file:
                pickle.dump(model, file)


    def train_val(self, save=True, val=True, plot=True, save_fig=True,
                  problem_name='', path='', fig_name='val'):
        """Perfomes training, validation and results printing.
        Saves models and validation results.

        Parameters
        ----------
        problem_name : str
            Dataset name and problem type
        path : str
            Path to the folder where models should be saved.
        fig_name : str
            Name of the figure to be saved.
        """

        self.train()
        print('')
        self.print_results()
        if val:
            print('\n'+'\033[1m'+'Bootstrap_632:'+'\033[0m')
            self.bootstrap_632()
            print('\n'+'\033[1m'+'Loo_cv:'+'\033[0m')
            self.loo_cv()
        if plot and val:
            self.plot_val(save_fig, fig_name)
        if save and val:
            print('\n''Saving results...')
            self.save_val_results(problem_name, path)
            self.save_models_pkl(problem_name, path)
            print('Done')
        elif save:
            print('\n''Saving results...')
            self.save_models_pkl(problem_name, path)
            print('Done')
