import numpy as np
import pandas as pd

def squash_array(x):
    x[x<0.0] == 0.0
    x[x>1.0] == 1.0
    return x

def squash_series(x):
    return x.apply(lambda x: max(x,0.0)).apply(lambda x: min(x,1.0))

def category_diff(cat1, cat2):
    different=False
    for k1,v1 in cat1.items():
        if k1 not in cat2.keys():
            print(f'{k1} not in cat2')
            different=True
        else:
            if not v1.equals(cat2[k1]):
                print(f'indices for {k1} different in cat2')
                different=True
    for k2,v2 in cat2.items():
        if k2 not in cat1.keys():
            print(f'{k1} not in cat1')
            different=True
        else:
            if not v2.equals(cat1[k2]):
                print(f'indices for {k2} different in cat1')
                different=True
    if not different:
        # print('categories match.')
        return True
    else:
        return False

def get_groups(df, groups, grouping):
    """Map data to an existing set of categories."""
    if grouping=='intersectional':
        group_ids = df.groupby(groups).groups
    elif grouping=='marginal':
        group_ids = {}
        for g in groups:
            grp = df.groupby(g).groups
            for k,v in grp.items():
                group_ids[(g,k)] = v
    return group_ids

def categorize(X, y, groups, grouping,
               n_bins=10,
               bins=None,
               alpha=0.0,
               gamma=0.0
              ):
    """Map data to an existing set of categories."""
    assert isinstance(X, pd.DataFrame), "X should be a dataframe"

    categories = None 

    if bins is None:
        if n_bins is None:
            n_bins = 10
        bins = np.linspace(float(1.0/n_bins), 1.0, n_bins)
        bins[0] = 0.0
    else:
        n_bins=len(bins)


    df = X[groups].copy()
    df.loc[:,'interval'], retbins = pd.cut(y, bins, 
                                           include_lowest=True,
                                           retbins=True
                                          )
    categories = {}
    group_ids = get_groups(df, groups, grouping)
    # group_ids = df.groupby(groups).groups

    min_grp_size = gamma*len(X) 
    min_cat_size = min_grp_size*alpha/n_bins
    for group, i in group_ids.items():
        # filter groups smaller than gamma*len(X)
        if len(i) <= min_grp_size:
            continue
        for interval, j in df.loc[i].groupby('interval').groups.items():
            # filter categories smaller than alpha*gamma*len(X)/n_bins
            if len(j) > min_cat_size:
                categories[group + (interval,)] = j
    return categories

nice_metrics = dict(
    subgroup_FNR_loss='FNR',
    subgroup_FPR_loss='FPR',
    subgroup_MSE_loss='Brier Score (MSE)',
    subgroup_positivity_loss='Positivity Rate',
    positivity='Positivity Rate',
    roc_auc_score='AUROC',
    average_precision_score='AUPRC',
    accuracy_score='Accuracy'
)