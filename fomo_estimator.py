from base_model import est as base_estimator
from fomo import FomoClassifier
from fomo.problem import LinearProblem, MLPProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
from sklearn.metrics import make_scorer
from pymoo.operators.crossover.sbx import SBX
from pymoo.termination.default import DefaultMultiObjectiveTermination
import fomo.metrics as metrics

est = FomoClassifier(
    estimator = base_estimator,
    accuracy_metrics=[make_scorer(metrics.FPR)],
    fairness_metrics=[metrics.subgroup_FNR_scorer],
    # algorithm = NSGA2(pop_size=50),
    algorithm = NSGA3(
        pop_size=64, 
        ref_dirs = get_reference_directions(
            "uniform", 
            2, 
            n_partitions=24
        ),
        crossover=SBX(eta=30, prob=0.2)
    ),
    verbose=True,
    problem_type=MLPProblem,
    checkpoint=True
)
est.n_jobs=min(64, est.algorithm.pop_size)
# est.n_jobs=1
termination = DefaultMultiObjectiveTermination(
    xtol=1e-8,
    cvtol=1e-6,
    ftol=0.0025,
    period=30,
    n_max_gen=1000,
    n_max_evals=100000
)