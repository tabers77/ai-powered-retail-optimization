from dataclasses import dataclass
from typing import Optional
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import model_selection as ms
import md_constants as constants


@dataclass
class SplitPolicy:
    """ Encapsulates configuration for building scikit-learn cross validation policies. """

    random_state: Optional[int] = None
    """ Used to seed random number generators when generating randomised splits """

    n_splits: int = 3
    """ Number of splits """

    policy_type: str = 'k_fold'
    """ Type of k_fold policy, one of 'k_fold', 'stratified_k_fold', 'repeated_k_fold',
            or 'repeated_stratified_k_fold'. """

    shuffle: bool = False
    """ Whether to shuffle. """

    n_repeats: int = 1
    """ Number of repeats. """

    def build(self):
        """ Builds the scikit-learn k fold policy, based on the attribute values."""
        if self.policy_type == 'k_fold':
            return ms.KFold(n_splits=self.n_splits, random_state=self.random_state, shuffle=self.shuffle)
        if self.policy_type == 'repeated_k_fold':
            return ms.RepeatedKFold(n_splits=self.n_splits,
                                    n_repeats=self.n_repeats,
                                    random_state=self.random_state)
        if self.policy_type == 'stratified_k_fold':
            return ms.StratifiedKFold(n_splits=self.n_splits, random_state=self.random_state, shuffle=self.shuffle)
        if self.policy_type == 'repeated_stratified_k_fold':
            return ms.RepeatedStratifiedKFold(n_splits=self.n_splits, random_state=self.random_state,
                                              n_repeats=self.n_repeats)
        raise ValueError(f'Unknown fold type: [{self.policy_type}]')

    # Provide some potentially useful defaults.
    @classmethod
    def kfold_default(cls):
        """ A default k_fold policy, using 3 splits, no shuffling. """

        return cls(random_state=constants.DEFAULT_SEED, n_splits=5, n_repeats=10, shuffle=True)

    @classmethod
    def repeated_kfold_default(cls):
        """ A default repeated k_fold policy, using 3 splits and 3 repeats. """

        return cls(n_repeats=3, policy_type='repeated_k_fold')

    @classmethod
    def stratified_kfold_default(cls):
        """ A default stratified k_fold policy using 3 splits, no shuffling. """

        return cls(policy_type='stratified_k_fold')

    @classmethod
    def repeated_stratified_kfold_default(cls):
        """ A default repeated stratified policy, with 3 repeats and 3 splits. """

        return cls(n_repeats=3, policy_type='repeated_stratified_k_fold')


# TODO : INVESTIGATE ROC AUC
class ScorersSpace:
    scorers_func = {'auc': accuracy_score,
                    # 'roc_auc': roc_auc_score,
                    'classification_report': classification_report,
                    'confusion_matrix': confusion_matrix

                    }
    scorers_name = {'auc': 'accuracy'}


def hold_out_scores(y_test, y_predicted, complete_score_set=False, scoring='auc'):
    scorers = ScorersSpace.scorers_func
    if not complete_score_set:
        score = scorers[scoring](y_test, y_predicted)
        return score

    else:
        scores = []
        for k, v in scorers.items():
            score = scorers[k](y_test, y_predicted)
            print(f'Score {k}', score)
            scores.append(score)

        return scores


def cross_validation(model, x, y, split_policy, scoring='auc'):
    scoring = ScorersSpace.scorers_name[scoring]
    scores = ms.cross_val_score(model, x, y, cv=split_policy,
                                scoring=scoring, n_jobs=constants.n_jobs, verbose=constants.verbose)
    return scores
