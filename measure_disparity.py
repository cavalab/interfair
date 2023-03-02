import numpy as np
import pandas as pd
import fire
import sklearn.metrics as sklearn_metrics
import metrics 
from utils import nice_metrics
import warnings
warnings.simplefilter('ignore')

def measure_disparity(
    dataset: str,
    save_file: str = 'df_fairness.csv'
):
    """Return prediction measures of disparity with respect to groups in dataset.

    Parameters
    ----------

    dataset: str
        A csv file storing a dataframe with one row per individual. Columns should include:

        1. `model prediction`: Model prediction (as a probability)
        2. `binary outcome`: Binary outcome (i.e. 0 or 1, where 1 indicates the favorable outcome for the individual being scored)
        3. `model label`: Model label
        4. `sample weights`: Sample weights
        5. additional columns are demographic data on protected and reference classes

    save_file: str, default: df_fairness.csv
        The name of the save file. 

    Outputs
    -------

    Overall Performance
        Predictive biases across data. 
    Subgroup Fairness Violations
        Deviations in performance for marginal and intersectional groups.
    Subgroups with largest violations
        Identifies groups experiencing the largest percent differences in performance according to each metric.  
    save_file: str, default df_fairness.csv
        Writes a csv file containing the fairness results.
    """
    print('reading in',dataset)
    df = pd.read_csv(dataset)
    required_cols = [
        'model prediction','binary outcome','model label','sample weights'
    ]
    for rc in required_cols:
        assert rc.lower() in df.columns, f'dataset must include a column labeled "{rc}".'

    demographics = [c for c in df.columns if c not in required_cols]
    print('demographic columns:',demographics)
    assert len(demographics) > 0, 'no demographic columns found.'

    weights = df['sample weights']

    soft_predictive_measures = [
        sklearn_metrics.roc_auc_score,
        sklearn_metrics.average_precision_score,
        metrics.positivity
    ]
    hard_predictive_measures = [
        metrics.FPR,
        metrics.FNR,
        sklearn_metrics.accuracy_score
    ]
    social_measures = [
        metrics.subgroup_FNR_loss, 
        metrics.subgroup_FPR_loss, 
        metrics.subgroup_MSE_loss, 
        metrics.subgroup_positivity_loss, 
        # metrics.multicalibration_loss, 
    ]
    y = df['binary outcome'].astype(int)
    y_pred = df['model label']
    y_pred_proba = df['model prediction']

    md_args = dict(
        index=False, 
        # tablefmt = 'fancy_outline'
        tablefmt = 'rounded_outline',
        stralign="right"
    )

    print(40*'=')
    print('Overall Performance')
    print(40*'=')
    print('\tMeasures of predictive bias on the whole population.')
    summary = {}
    for pm in soft_predictive_measures:
        name = nice_metrics.get(pm.__name__,pm.__name__)
        summary[name] = pm(y, y_pred_proba)
    for pm in hard_predictive_measures:
        name = nice_metrics.get(pm.__name__,pm.__name__)
        summary[name] = pm(y, y_pred)
    df_summary = pd.DataFrame(summary, index=['value'])
    print(df_summary.round(3).to_markdown(**md_args))


    print(40*'=')
    print('Subgroup Fairness Violations')
    print(40*'=')
    print('\tMeasures the deviation in performance for marginal and intersectional groups.')
    print('\tNote that these deviation are weighted by group prevalence to produce stable estimates when sample sizes are small.')
    X_protected = df[demographics]
    frames = []
    for sm in social_measures:
        for grouping in ['marginal','intersectional']:
            # print(sm.__name__)
            result, max_loss, max_group = sm(
                y, y_pred_proba, X_protected, 
                weights=weights,
                grouping=grouping
            )
            result['metric'] = nice_metrics.get(sm.__name__,sm.__name__)
            result['grouping'] = grouping
            frames.append(result)

    df_fairness = (
        pd.concat(frames)
        .pivot(columns=['metric'], values=['signed_value'])
    )
    df_fairness.columns = df_fairness.columns.get_level_values(1)
    # get worst group violations
    worst_indices = {}
    for col in df_fairness.columns:
        if col == 'Positivity Rate':
            worst_group = df_fairness[col].argmin() 
        else:
            worst_group = df_fairness[col].argmax() 
        worst_indices[col] = df_fairness.iloc[worst_group]._name

    df_tbl = df_fairness.round(3) #.astype(str)
    for col,idx in worst_indices.items():
        df_tbl.loc[idx,col] = '**'+df_tbl.loc[idx,col].astype(str)
    print(df_tbl.reset_index().to_markdown(**md_args))
    
    # text of worst groups
    df_raw = (
        pd.concat(frames)
        .pivot(columns=['metric'], values=['raw_value_pct'])
    )
    df_raw.columns = df_raw.columns.get_level_values(1)
    print('Subgroups with Largest Deviations')
    print(20*'-')
    for col,idx in worst_indices.items():
        print(col)
        print(10*'-')
        print('-','Subgroup:',
        ','.join([f'{k}={v}' for k,v in zip(df_fairness.index.names,idx) if v != '  any  '])
        )
        pct_diff = df_raw.loc[idx,col]
        higher = df_fairness.loc[idx,col] > 0
        print(f'- {col} is {pct_diff:.1f} % {"higher" if higher else "lower"} among this'
        ' group than the population.\n'
        )

    print('saving results to',save_file)
    df_fairness.reset_index().to_csv(save_file, index=False)

if __name__ == '__main__':
    fire.Fire(measure_disparity)