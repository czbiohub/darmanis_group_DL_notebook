# libraries
import pandas as pd
from pandas.api.types import CategoricalDtype, is_categorical_dtype
import numpy as np
import string
import types
import scanpy.api as sc
import anndata as ad
from plotnine import *
import plotnine
import scipy
from scipy import sparse, stats
import glob
import more_itertools as mit
import tqdm
import pickle
import multiprocessing
import itertools
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import typing
import random
from adjustText import adjust_text
import sys

# classes
class SklearnWrapper:
    """
    class to handle sklearn function piped inline with pandas
    """
    def __init__(self, transform: typing.Callable):
        self.transform = transform

    def __call__(self, df):
        transformed = self.transform.fit_transform(df.values)
        return pd.DataFrame(transformed, columns=df.columns, index=df.index)

# functions
def imports():
    """
    print module names and versions 
    ref: https://stackoverflow.com/questions/20180543/how-to-check-version-of-python-modules
    
    input: none
    output: print to std out
    """
    for name, val in globals().items():
        if isinstance(val, types.ModuleType):
            if val.__name__ not in ['builtins']:
                try:
                    print (f'{val.__name__}:', val.__version__)
                except:
                    pass
                    
def create_adata (pre_adata):
    """
    Creates adata obj from raw data (rows=gene_names, col=cell_id)
    
    Input: raw expression data in pd df
    Output: adata obj
    """

    
    print('Ingest raw data...')

    # pd df to np array
    array_adata = pre_adata.values

    # extract obs and var
    obs = pre_adata.columns.tolist()
    gene_names = pre_adata.index.tolist()
    var = pd.DataFrame({'gene_symbols':gene_names})
    
    # create ad obj
    adata = ad.AnnData(X=array_adata).T
    adata.X = sparse.csr_matrix(adata.X)
    adata.var_names = gene_names
    adata.obs_names = obs
    
    return adata
    
def append_anno (adata, anno, anno_dict):
    """
    Add annotations of choice from annotation file
    
    input = adata obj + dictionary of label and column name (with respect to annotation df) + anno pd df
    output = updated adata obj
    
    """

    
    print('Append annotations...')
    
    anno = anno
    anno_dict = anno_dict
    
    # append metadata of choice
    for key,value in anno_dict.items():
        adata.obs[value] = anno[key].values

def remove_ercc (adata):
    """
    Remove ercc spike-in
    
    Input: adata obj
    Output: updated adata obj
    """
    
    print('Remove ERCC genes...')
    
    gene_names = adata.var_names.tolist()
    ERCC_hits = list(filter(lambda x: 'ERCC' in x, gene_names))
    adata = adata[:, [x for x in gene_names if not (x in ERCC_hits)]]
    
    return adata

def technical_filters (adata, min_genes=500,min_counts=50000,min_cells=3):
    """
    remove cells/genes based on low quality
    
    input: adata
    output: updated adata obj 
    """

    print('Remove low-quality cells/genes...')

    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_cells(adata, min_counts=min_counts)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    
def prepare_dataframe(adata, var_names, groupby=None, use_raw=None, log=False, num_categories=7):
    """
    ### FROM scanpy ###
    
    Given the anndata object, prepares a data frame in which the row index are the categories
    defined by group by and the columns correspond to var_names.
    Parameters
    ----------
    adata : :class:`~anndata.AnnData`
        Annotated data matrix.
    var_names : `str` or list of `str`
        `var_names` should be a valid subset of  `adata.var_names`.
    groupby : `str` or `None`, optional (default: `None`)
        The key of the observation grouping to consider. It is expected that
        groupby is a categorical. If groupby is not a categorical observation,
        it would be subdivided into `num_categories`.
    log : `bool`, optional (default: `False`)
        Use the log of the values
    use_raw : `bool`, optional (default: `None`)
        Use `raw` attribute of `adata` if present.
    num_categories : `int`, optional (default: `7`)
        Only used if groupby observation is not categorical. This value
        determines the number of groups into which the groupby observation
        should be subdivided.
    Returns
    -------
    Tuple of `pandas.DataFrame` and list of categories.
    """
    from scipy.sparse import issparse
#     sanitize_anndata(adata)
    if use_raw is None and adata.raw is not None: use_raw = True
    if isinstance(var_names, str):
        var_names = [var_names]

    if groupby is not None:
        if groupby not in adata.obs_keys():
            raise ValueError('groupby has to be a valid observation. Given value: {}, '
                             'valid observations: {}'.format(groupby, adata.obs_keys()))

    if use_raw:
        matrix = adata.raw[:, var_names].X
    else:
        matrix = adata[:, var_names].X

    if issparse(matrix):
        matrix = matrix.toarray()
    if log:
        matrix = np.log1p(matrix)

    obs_tidy = pd.DataFrame(matrix, columns=var_names)
    if groupby is None:
        groupby = ''
        categorical = pd.Series(np.repeat('', len(obs_tidy))).astype('category')
    else:
        if not is_categorical_dtype(adata.obs[groupby]):
            # if the groupby column is not categorical, turn it into one
            # by subdividing into  `num_categories` categories
            categorical = pd.cut(adata.obs[groupby], num_categories)
        else:
            categorical = adata.obs[groupby]

    obs_tidy.set_index(categorical, groupby, inplace=True)
    categories = obs_tidy.index.categories

    return categories, obs_tidy

def regress(x, y, predictor, response, fit_intercept = False):
    """
    wrapper for linear regression
    
    input: 1D array of shape (-1, 1)
    output: fit and residuals
    """
    
    # Model initialization
    regression_model =  LinearRegression(fit_intercept=fit_intercept)
    # Fit the data(train the model)
    regression_model.fit(x, y)
    # Predict
    y_predicted = regression_model.predict(predictor).reshape((-1,1))
    y_residuals = response - y_predicted
    # coef
    m = regression_model.coef_[0][0]
    
    return y_predicted.flatten(), y_residuals.flatten(), m

def adata_DE_pairwise(input_adata, 
                      groupby, 
                      target_1, 
                      target_2, 
                      method = 'wilcoxon',
                      corr_method = 'benjamini-hochberg'
                     ):
    """
    Wrapper for scanpy DE tests. Two-sided.
    
    Input: adata, groupby variable, comparison labels, test, mutliple hypothesis procedure
    Output: dataframe of gene, log2fc, pval, adj_pval
    """
    
    n_genes=len(input_adata.var_names)
    sc.tl.rank_genes_groups(input_adata, 
                            groupby=groupby, 
                            groups=[target_1],
                            reference=target_2,
                            method=method,
                            n_genes = n_genes,
                            corr_method = corr_method
                           )
    genes = [x[0] for x in input_adata.uns['rank_genes_groups']['names']]
    log2change = [x[0] for x in input_adata.uns['rank_genes_groups']['logfoldchanges']]
    pvals = [x[0] for x in input_adata.uns['rank_genes_groups']['pvals']]
    pvals_adj = [x[0] for x in input_adata.uns['rank_genes_groups']['pvals_adj']]
    
    results_df = pd.DataFrame({
        'gene':genes,
        'log2change':log2change,
        'pvals':pvals,
        'pvals_adj':pvals_adj
    })
    
    return results_df


# Global variables

color_code_dict = {'dendritic':'#b99abf',
                   'cyc_dendritic':'#9abfb9',
                   'eccrine':'#CBB7E3',
                   'cyc_eccrine':'#b7e3cb',
                   'krt':'#E9E1F2',
                   'cyc_krt':'#e1f2e9',
                   'mast':'#AE90C2',
                   'T_cell':'#5F3D68',
                   'mel':'#000000',
                   'cyc_mel':'#999999',
                   'cutaneous_mel':'#FF0000',
                   'cutaneous':'#FF0000',
                   'leg':'#FF0000',
                   'arm':'#FF0000',
                   'acral_mel':'#0000FF',
                   'acral':'#0000FF',
                   'palm':'#0000FF',
                   'sole':'#0000FF',
                   'foreskin_mel':'#FFA500',
                   'foreskin':'#FFA500',
                   'dark_foll_mel':'#003300',
                   'light_foll_mel':'#99cc99',
                   'follicular':'#008000',
                   'hair_follicle':'#008000',
                   'fet_cutaneous_mel':'#ff4c4c',
                   'adt_cutaneous_mel':'#b20000',
                   'shallow_regime':'#b20000',
                   'steep_regime':'#00b2b2',
                   'fet_acral_mel':'#4c4cff',
                   'adt_acral_mel':'#0000b2',
                   'neo_foreskin_mel':'#FFA500',
                   'fet_dark_foll_mel':'#003300',
                   'fet_light_foll_mel':'#99cc99',
                   'fet':'#dbc2a9',
                   'neo':'#c09569',
                   'adt':'#a5682a',
                   'NTRK2+/HPGD+':'#474747',
                   'NTRK2-/HPGD-':'#DDDDDD',
                   'NTRK2+/HPGD-':'#0000FF',
                   'NTRK2-/HPGD+':'#FF0000',
                   'black':'#000000',
                   'grey':'#D3D3D3',
                   'melanoma':'#935aff',
                   'mel':'#935aff',
                   'follicular_like':'#6514ff',
                   'adult_interfollicular':'#ff1439',
                   'follicular_low':'#ff1439',
                   'interfoll_mel':'#ff1439',
                   'neonatal_interfollicular':'#ffda14',
                   'fetal_interfollicular':'#1439ff',
                   'fetal_follicular':'#39ff14',
                   'follicular_high':'#39ff14',
                   'light_foll_mel':'#39ff14',
                   'dark_foll_mel':'#93ba8b',
                   'norm':'#000000',
                   'cluster_1':'#ff1439',
                   'cluster_0':'#ffda14',
                   'cluster_2':'#39ff14',
                  }
heatmap_cmap = 'jet'