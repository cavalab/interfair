from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_halving_search_cv # noqa
from sklearn.model_selection import HalvingGridSearchCV
import numpy as np
from tempfile import mkdtemp
cachedir = mkdtemp()

base_model = LogisticRegression(n_jobs=1, solver='saga',penalty='l1')
# base_model = RandomForestClassifier(n_jobs=1)

categorical_features = ['insurance','ethnicity']  
numeric_features = [
    'temperature',
    'heartrate',
    'resprate',
    'o2sat',
    'sbp',
    'dbp',
    'pain',
    'acuity',
    'prev_adm'
]
print('categorical features:',categorical_features)
print('numeric features:',numeric_features)

numeric_transformer = make_pipeline(
    SimpleImputer(strategy="median"), 
    StandardScaler()
    )

preprocessor = ColumnTransformer(
    [
        ("num", numeric_transformer, numeric_features),
        # (
        #     "cat",
        #     OneHotEncoder(
        #         handle_unknown="ignore", 
        #         sparse_output=False
        #     ),
        #     categorical_features,
        # ),
    ],
    verbose_feature_names_out=False,
    remainder='passthrough'
)
est = make_pipeline(preprocessor, base_model)

from xgboost import XGBRFClassifier 

est = XGBRFClassifier(n_jobs=1)