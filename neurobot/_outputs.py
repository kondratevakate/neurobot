


def print_results_(clf_grid_dict, save_plot_to=None):
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
    
    # draw graph
    width = 0.9
    for i in results.index:
        plt.bar(i, results.loc[i, "mean"], width, yerr=results.loc[i, "std"], label=results.loc[i, "classifier"])
    plt.xticks(range(results.shape[0]), results.loc[:, "classifier"])
    plt.axis(ymin=0.0, ymax=1.0)
    if save_plot_to is not None:
        plt.savefig(save_plot_to)
    plt.show()
    
    cnt=0
    fig, axs = plt.subplots(len(results["classifier"]), figsize=(12,25))
    for clf in results["classifier"]:
        graph_mean = []
        graph_std = []
        j=0
        for n in range(25):
            #print('split{}_test_score'.format(n), grids[clf].cv_results_['split{}_test_score'.format(n)])
            graph_mean.append(clf_grid_dict[clf].cv_results_['split{}_test_score'.format(n)].mean())
            graph_std.append(clf_grid_dict[clf].cv_results_['split{}_test_score'.format(n)].std())
        for i in range(5):
            axs[cnt].bar(range(j, j+5), [graph_mean[m] for m in range(j, j+5)], yerr=[graph_std[m] for m in range(j, j+5)])
            j+=5
        axs[cnt].set_xticks(np.linspace(2,22,5))
        axs[cnt].set_xticklabels(range(1,6))
        axs[cnt].set_title(results["classifier"][cnt].upper())
        for ax in axs.flat:
            ax.set(xlabel='5 folds with 5 iterations for each', ylabel='Mean score for each iteration')
        cnt+=1
    plt.show()
    
    print("Best model: ")
    clf = results.loc[results["mean"].argmax(), "classifier"]
    print(clf)
    print("\n".join(
            [param + " = " + str(best_value) for param, best_value in clf_grid_dict[clf].best_params_.items()]))
