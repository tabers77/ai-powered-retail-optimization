from sklearn import model_selection as ms

from md_configs import setUp
from md_helpers import SplitsHandler


class HyperOpt:
    def __init__(self):
        pass

    @staticmethod
    def grid_search_hyperopt(grid_search_param_grid: dict,
                             df=None,
                             x=None,
                             y=None,
                             model_name=None,
                             model=None,
                             evaluation_metric="accuracy",
                             n_jobs=-1,
                             verbose=3,
                             n_folds=3,
                             n_repeats=3,
                             k_fold_method='k_fold',
                             grid_search_method='randomized',
                             random_state=setUp['DEFAULT_SEED']):

        if df is not None and (x is None and y is None):
            splitter = SplitsHandler()
            x, y = splitter.get_x_y_from_df(df, setUp['target_variable'])
        elif df is not None and (x is not None and y is not None):
            raise ValueError('Check the inputs!!!')

        k_fold_methods_dict = {'k_fold': ms.KFold(n_splits=n_folds, shuffle=True),
                               'repeated_k_fold': ms.RepeatedKFold(n_splits=n_folds, n_repeats=n_repeats,
                                                                   random_state=random_state),
                               'stratified_k_fold': ms.StratifiedKFold(n_splits=n_folds, shuffle=True,
                                                                       random_state=random_state),
                               'repeated_stratified_k_fold': ms.RepeatedStratifiedKFold(n_splits=n_folds,
                                                                                        n_repeats=n_repeats,
                                                                                        random_state=random_state)}

        current_kfold = k_fold_methods_dict[k_fold_method]

        grid_search_dict = {
            'randomized': ms.RandomizedSearchCV(model, grid_search_param_grid, scoring=evaluation_metric, n_jobs=n_jobs,
                                                cv=current_kfold,  # this line will be modified or removed
                                                verbose=verbose),
            'gridsearch': ms.GridSearchCV(model, grid_search_param_grid, scoring=evaluation_metric, n_jobs=n_jobs,
                                          cv=current_kfold,  # this line will be modified or removed
                                          verbose=verbose)}  # this does not save results during the process

        current_grid_search = grid_search_dict[grid_search_method]

        grid_result = current_grid_search.fit(x, y)
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

        scores = {model_name: (grid_result.best_score_, None)}
        tuned_model = model.set_params(**grid_result.best_params_)

        return scores, grid_result.best_params_, tuned_model
