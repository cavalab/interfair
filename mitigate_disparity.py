import pandas as pd

from fomo import FomoClassifier
from fomo.problem import LinearProblem, MLPProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
import fomo.metrics as metrics
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer
from base_model import est as base_estimator
from sklearn.utils import resample
from pymoo.operators.crossover.sbx import SBX
import pickle

def mitigate_disparity(
    dataset: str,
    protected_features: list[str]
):
    """
    “mitigate_disparity.py” takes in a model development dataset (training and test datasets) that your algorithm has not seen before and generates a new, optimally fair/debiased model that can be used to make new predictions.

    Parameters
    ----------
    dataset: str
      A csv file storing a dataframe with one row per individual. Columns should include:
      1. `binary outcome`: Binary outcome (i.e. 0 or 1, where 1 indicates the favorable outcome for the individual being scored)
      2. `sample weights`: Sample weights. These are ignored. 
      3. All additional columns are treated as features/predictors.

    protected_features: list[str]
        The columns of the dataset over which we wish to control for fairness.

    Returns
    -------

    (1)  The fair/debiased model object, taking the form of a sklearn-style python object with the following functions accessible:

    (i)    .fit() – trains the model

    (ii)   .predict() / .predict_proba() – makes predictions using new data

    (iii)  .transform() – filters or modifies input data, if applicable

    (2)  [Optional] graphics/visualization, useful formatted output
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
    

    est = FomoClassifier(
        estimator = base_estimator,
        accuracy_metrics=[make_scorer(metrics.FPR)],
        fairness_metrics=[metrics.subgroup_FNR_scorer],
        # algorithm = NSGA2(pop_size=100),
        algorithm = NSGA3(
            pop_size=100, 
            ref_dirs = get_reference_directions(
                "uniform", 2, 
                n_partitions=10),
            crossover=SBX(eta=30, prob=0.2)
        ),
        verbose=True,
        problem_type=LinearProblem,
        n_jobs=8
    )
    Xtrain,ytrain = resample(Xtrain,ytrain, n_samples=10000)
    est.fit(
        Xtrain,
        ytrain,
        protected_features=list(protected_features), 
        termination=('n_gen',50),
    )
    with open( 'estimator.pkl', 'wb') as of:
        pickle.dump(est, of)
    return est

import fire    
if __name__ == '__main__':
  fire.Fire(mitigate_disparity)