import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import pickle
import fomo_estimator

def mitigate_disparity(
    dataset: str,
    protected_features: list[str],
    starting_point: str|None = None,
    save_file: str = 'estimator.pkl'
):
    """
    “mitigate_disparity.py” takes in a model development dataset (training and test datasets) that your algorithm has not seen before and generates a new, optimally fair/debiased model that can be used to make new predictions.

    Parameters
    ----------
    dataset: str
        A csv file storing a dataframe with one row per individual. 
        Columns should include:
        1. `binary outcome`: Binary outcome (i.e. 0 or 1, where 1 indicates the 
        favorable outcome for the individual being scored)
        2. `sample weights`: Sample weights. These are ignored. 
        3. All additional columns are treated as features/predictors.
    protected_features: list[str]
        The columns of the dataset over which we wish to control for fairness.
    starting_point : str | None
        Optionally start from a checkpoint file with this name.
    save_file: str, default: estimator.pkl
        The name of the saved estimator. 

    Returns
    -------
    estimator.pkl: file containing sklearn-style Estimator
        Saves a fair/debiased model object, taking the form of a sklearn-style python object.

    """

    print('dataset:',dataset)
    print('protected_features:',protected_features)

    df = pd.read_csv(dataset, index_col=False)
    X = df.drop(columns=['binary outcome', 'sample weights'],
                axis=1,
                errors='ignore'
                )
    y = df['binary outcome']
    est = fomo_estimator.est

    est.fit(
        X,
        y,
        protected_features=list(protected_features), 
        termination=fomo_estimator.termination,
        starting_point=starting_point,
        save_history=True,
        checkpoint=True
    )
    print('saving estimator to',save_file,'...')
    with open(save_file, 'wb') as of:
        pickle.dump(est, of)
    print('done.')

import fire    
if __name__ == '__main__':
  fire.Fire(mitigate_disparity)