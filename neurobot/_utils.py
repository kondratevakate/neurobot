  
def repeated_cross_val_predict_proba(estimator, X, y, n_objects, 
                                     cv, pos_label = None, file = None):
    if pos_label is None:
        y_enc = pd.Series(LabelEncoder().fit_transform(y), index = y.index)
    else:
        y_enc = pd.Series(y == pos_label, dtype = int)
    predictions = [[] for i in range(n_objects)]
    for idx_tr, idx_te in tqdm(cv.split(X, y_enc)):
        estimator.fit(X.iloc[idx_tr], y_enc.iloc[idx_tr])
        pred_te = np.array(estimator.predict_proba(X.iloc[idx_te]), dtype = float)
        for i, idx in enumerate(idx_te):
            predictions[X.index[idx]].append(pred_te[i, 1])
        
    predictions = pd.DataFrame(predictions)
    if file is not None:
        predictions.to_csv(file)
    return predictions


def get_feature_sets_on_cross_val(X, y, model, cv):
    feature_sets = []
    for idx_tr, idx_te in tqdm(cv.split(X, y)):
        X_tr = X.loc[X.index[idx_tr]]
        y_tr = y.loc[X.index[idx_tr]]
        y_te = y.loc[X.index[idx_te]]
        feature_sets.append(get_features(X_tr, y_tr, model))
    return feature_sets


def get_features(X, y, model):
    model.fit(X, y)
    dim_reduction = model.named_steps["dim_reduction"]
    features_idx = dim_reduction.get_support()
    features = X.columns[features_idx].tolist()
    return features


def plot_roc_curve(y, probas, average_repeats=False, show=True):
    if average_repeats:
        y_true = y
        y_score = probas.mean(axis=1)
    else:
        n_repeats = probas.shape[1]
        y_true = pd.Series(np.tile(y, (n_repeats)), dtype=int)
        y_score = probas.values.T.reshape(-1, 1)
    fpr, tpr, t = roc_curve(y_true=y_true, y_score=y_score)
    
    if show:
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr)
        plt.xlabel("False Positive rate", fontsize=14)
        plt.ylabel("True Positive rate", fontsize=14)
        plt.show()
        print("auc =", roc_auc_score(y_true, y_score))        
    return fpr, tpr, t


def get_fpr_fnr(fpr, tpr, fix_fpr_l=[0.1, 0.15, 0.2, 0.3]):
    fnr_l = []
    for fix_fpr in fix_fpr_l:
        fnr_l.append(1 - tpr[fpr <= fix_fpr][-1])
    fpr_fnr_table = pd.DataFrame(
        np.column_stack((fix_fpr_l, fnr_l)), columns=["False Positive rate (fixed)", "False Negative rate"]
    )
    display(fpr_fnr_table)
    return fpr_fnr_table


def get_incorrectly_classified(labels, y, probas, idx, fpr, t, fix_fpr_l=[0.1, 0.15, 0.2, 0.3], file=None, show=True):
    columns = ["False Positive rate (fixed)", "Threshold", "False Positives indexes", "False Negatives indexes"]
    t_l = []
    false_0 = []
    false_1 = []
    for fix_fpr in fix_fpr_l:
        fix_t = t[fpr <= fix_fpr][-1]
        t_l.append(fix_t)
        labels_t = probas > fix_t
        labels_t = pd.Series(labels_t.values.ravel())
        false_0.append(
            ", ".join(
                list(
                    labels.loc[(probas[idx][np.logical_and(labels_t[idx] == 0, y == 1)]).index, "patient_number"]
                )
            )
        )
        false_1.append(
            ", ".join(
                list(
                    labels.loc[(probas[idx][np.logical_and(labels_t[idx] == 1, y == 0)]).index, "patient_number"]
                )
            )
        )
              
    t_l = np.array(t_l)
    false_0 = np.array(false_0)
    false_1 = np.array(false_1)
    
    res = pd.DataFrame(np.column_stack((fix_fpr_l, t_l)), columns=columns[:2])
    res["False Positives indexes"] = false_1
    res["False Negatives indexes"] = false_0
    
    if file is not None:
        res.to_csv(file)
        
    if show:
        display(res)
    return res