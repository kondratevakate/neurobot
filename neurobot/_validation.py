 def loo_cv(self):
        """Performs Leave-One-Out cross validation.
        """

        print (self.problem_name)
        self.loo_results = []
        for k in range(1, len(self.grid)):
            start_time = time.time()
            best_model = self.grid[k].best_estimator_
            loo = LeaveOneOut()
            loo.get_n_splits(self.X)
            predict = []
            
            for train_index, test_index in loo.split(self.X):
                X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
                y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]
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
                self.classifiers[k].upper()  + ': ',
                ' acc', np.round((tpr + tnr) / 2, 2), 
                ' tpr', np.round(tpr, 2),
                ' tnr', np.round(tnr, 2),
                ' time', end_time
            )
            
def bootstrap_632(self):
        """Performs bootstrap validation.
        """

        print (self.problem_name)
        self.bootstrap_results = []
        for k in range(1, len(self.grid)):
            start_time = time.time()
            scores = bootstrap_point632_score(
                self.grid[k].best_estimator_, self.X.values,
                self.y.values, n_splits=1000,
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
                self.classifiers[k].upper(),
                ' acc: %.2f%%' % (100*acc), 
                ' 95%% Confidence interval: [%.2f, %.2f]' % \
                    (100*lower, 100*upper),
                ' time', end_time
            )
        