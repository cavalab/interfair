from base_model import est as base_estimator
from fomo import FomoClassifier
from fomo.problem import MLPProblem
import fomo.metrics as metrics
from pymoo.algorithms.moo.nsga2 import NSGA2
from sklearn.metrics import make_scorer
from pymoo.operators.crossover.sbx import SBX
from pymoo.termination.default import DefaultMultiObjectiveTermination

est = FomoClassifier(
    estimator = base_estimator,
    accuracy_metrics=[make_scorer(metrics.FPR)],
    fairness_metrics=[metrics.subgroup_FNR_scorer],
    algorithm = NSGA2(pop_size=50),
    verbose=True,
    problem_type=MLPProblem,
    checkpoint=False
)
est.n_jobs=min(64, est.algorithm.pop_size)
# est.n_jobs=1
termination = DefaultMultiObjectiveTermination(
    xtol=1e-8,
    cvtol=1e-6,
    ftol=0.0025,
    period=30,
    n_max_gen=100,
    n_max_evals=100000
)