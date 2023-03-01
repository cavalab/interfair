import pandas as pd
from pymoo.termination.default import DefaultMultiObjectiveTermination
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import pickle

# from pymoo.core.callback import Callback
# from pymoo.problems import get_problem
# from pymoo.optimize import minimize
# from pymoo.visualization.pcp import PCP
# from pyrecorder.recorder import Recorder
# from pyrecorder.writers.streamer import Streamer

import fomo_estimator

def mitigate_disparity(
    dataset: str,
    protected_features: list[str],
    starting_point: str|None = None
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
    Xtrain,Xtest, ytrain,ytest = train_test_split(
        X,y,
        stratify=y,
        random_state=42,
        test_size=0.5
    )
    Xtrain,ytrain = resample(Xtrain,ytrain, n_samples=10000)
    
    est = fomo_estimator.est

    termination = DefaultMultiObjectiveTermination(
        xtol=1e-8,
        cvtol=1e-6,
        ftol=0.0025,
        period=30,
        n_max_gen=10,
        n_max_evals=100000
    )
    est.fit(
        Xtrain,
        ytrain,
        protected_features=list(protected_features), 
        termination=termination,
        starting_point=starting_point,
        save_history=True
    )
    print('saving estimator to estimator.pkl...')
    with open( 'estimator.pkl', 'wb') as of:
        pickle.dump(est, of)
    print('done.')

import fire    
if __name__ == '__main__':
  fire.Fire(mitigate_disparity)