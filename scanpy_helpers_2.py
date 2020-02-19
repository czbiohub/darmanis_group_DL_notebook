# FUNCTIONS
from collections import defaultdict
import multiprocessing
# linear regression between cancer
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.metrics import mean_squared_error, r2_score, classification_report
from sklearn.cluster import AgglomerativeClustering
from scipy.interpolate import LSQUnivariateSpline, UnivariateSpline
from math import sqrt, pi

# base
import pandas as pd
from pandas.api.types import CategoricalDtype, is_categorical_dtype
import numpy as np
from scipy import sparse, stats
from scipy.stats import ttest_ind, ranksums, variation, pearsonr
import warnings
import scipy.stats as ss
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.cluster import AgglomerativeClustering, KMeans
from scipy.cluster import hierarchy
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, jaccard_similarity_score
from IPython.core.display import HTML
import numpy.ma as ma # masking package
import statsmodels.api as sm
import subprocess, os, sys, mygene, string, glob, typing, random, pickle, tqdm, itertools
import s3fs
import boto3
from lifelines.statistics import logrank_test


# sc analysis
import scanpy.api as sc
import anndata as ad
import gseapy as gp

#plotting
from plotnine import *
import plotnine
import matplotlib as mp
import matplotlib.pyplot as plt
from adjustText import adjust_text
from matplotlib_venn import venn3, venn3_circles, venn2


# uniprot api
from bioservices import UniProt
u = UniProt()

class SklearnWrapper:
    def __init__(self, transform: typing.Callable):
        self.transform = transform

    def __call__(self, df):
        transformed = self.transform.fit_transform(df.values)
        return pd.DataFrame(transformed, columns=df.columns, index=df.index)

def single_mwu(args):    
    # inputs
    gene_val, df1, df2, alt = args
    
    try:
        stat, pval = stats.mannwhitneyu(df1[gene_val].values,
                                        df2[gene_val].values, 
                                        alternative = alt
                                       )
        logfc = np.log2(np.mean(df1[gene_val].values)+1)-np.log2(np.mean(df2[gene_val].values)+1)
    except:
        stat = np.nan
        pval = np.nan
        logfc = np.nan
    return (stat, pval, logfc)

def parallel_mwu(genes, df1, df2, method='two-sided', ncores=1):
    
    p = multiprocessing.Pool(processes=ncores)
    try:
        mwu_tuples = p.map(single_mwu, 
                                zip(genes,
                                    [df1]*len(genes),
                                    [df2]*len(genes),
                                    [method]*len(genes)
                                   ), 
                                chunksize=100)
    finally:
        p.close()
        p.join()

    stat_list = [x[0] for x in mwu_tuples]
    pval_list = [x[1] for x in mwu_tuples]
    logfc_list = [x[2] for x in mwu_tuples]
    return (stat_list, pval_list, logfc_list)
    

def single_ranks(args):    
    # inputs
    ## df1 = single cell df 
    df1_row, df1, df2 = args
    
    # filter by nonzero expressing genes in single cell and rank
    return_genes = []
    return_ranks = []
    for df2_row in range(len(df2)):
        keep_idx = df1.iloc[df1_row,:].values > 0
        gene_vals = np.array(df1.columns)[keep_idx].tolist()
        df1_vals = stats.rankdata(df1.iloc[df1_row,:].values[keep_idx])
        df2_vals = stats.rankdata(df2.iloc[df2_row,:].values[keep_idx])
        rank_diff_vals = abs(df1_vals - df2_vals).tolist()
        
        return_genes = return_genes + gene_vals
        return_ranks = return_ranks + rank_diff_vals
    
    return (return_genes, return_ranks)

def parallel_ranks(df1, df2, ncore = 1):

    rank_diff_dict = defaultdict(list)
    p = multiprocessing.Pool(processes=ncore)
    jobs_list = [x for x in range(len(df1))]

    try:
        rank_tuples = p.map(single_ranks, 
                            zip(jobs_list,
                                [df1]*len(jobs_list),
                                [df2]*len(jobs_list)
                               ), 
                                chunksize=100)
    finally:
        p.close()
        p.join()
        
    for x in rank_tuples:
        genes = x[0]
        ranks = x[1]
        for gene,rank in zip(genes, ranks):
            rank_diff_dict[gene].append(rank)
            
    for key,value in rank_diff_dict.items():
        rank_diff_dict[key] = np.median(value) 
        
    genes = [key for key,value in rank_diff_dict.items()]
    rank_diffs = [value for key,value in rank_diff_dict.items()]
            
    return (genes, rank_diffs)

def parallel_paired_spearman(args):
        ### needs refactor ###
    # reference df  MUST be single cell in order to filter on nonzero expressing genes
    
    # inputs
    ref_idx, ref_df, cross_df, nonzero_only = args

    # slice df
    ref_slice = ref_df.iloc[ref_idx,:]
    if nonzero_only == True:
        ref_slice = ref_slice[ref_slice > 0]
    else:
        pass
    ref_values = ref_slice.values
    ref_cols = ref_slice.index.tolist()
    cross_slice = cross_df.loc[:,ref_cols]
    
    stat_list = []
    pval_list = []
    for i in range(len(cross_slice)):
        cross_values = cross_slice.iloc[i,:].values
        try:
            stat, pval = stats.spearmanr(ref_values,cross_values)
        except:    
            stat = pval = np.nan
        stat_list.append(stat)
        pval_list.append(pval)

    return (stat_list, pval_list)

def geneset_lookup(glist, 
                   outdir = '/home/ubuntu/data/enrichr_kegg',
                   gene_sets = ['KEGG_2016',
                                'GO_Molecular_Function_2018',
                                'GO_Biological_Process_2018',
                                'GO_Cellular_Component_2018',
                                'WikiPathways_2016'
                                ]
                  ):
    import gseapy as gp
    
    # API call
    enr = gp.enrichr(gene_list = glist,
                            description='test',
                            gene_sets=gene_sets,
                            outdir=f'{outdir}', 
                            cutoff=0.5)

    # output results
    return enr.results.sort_values('Adjusted P-value')

def pca_logistic(pred, res):

    res = res.reshape(-1,1)

    if len(np.unique(res)) == 1:
        acc = 0
    else:
        X_train, X_test, y_train, y_test = train_test_split(pred,
                                                            res,
                                                            test_size=0.33, 
                                                            random_state=42)
        # accurcy
        clf = LogisticRegression(multi_class='auto')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc  = f1_score(y_true = y_test, 
                        y_pred = y_pred, 
                        average = 'micro')

    return acc

def zero_fraction(x):
    value = (len(x)-np.count_nonzero(x))/len(x)
    value_str = '{}'.format(round(value,2))
    return value

def regress(x, y, predictor, fit_intercept = False):
    # requires 1D array of shape (-1, 1)
    # returns fit and residuals residuals
    
    # Model initialization
    regression_model =  LinearRegression(fit_intercept=fit_intercept)
    # Fit the data(train the model)
    regression_model.fit(x, y)
    # Predict
    y_predicted = regression_model.predict(predictor).reshape((-1,1))
    y_residuals = predictor - y_predicted
    
    return y_predicted.flatten(), y_residuals.flatten()

def get_s3path_list(bucket, prefix, suffix):
    #     bucket = 'darmanis-group'
    #     prefix = 'singlecell_lungadeno/rawdata/fastqs'
    #     suffix = 'fastq.gz'

    client = boto3.client('s3')
    paginator = client.get_paginator('list_objects')
    operation_parameters = {'Bucket': bucket,
                            'Prefix': prefix}
    page_iterator = paginator.paginate(**operation_parameters)
    paths = ['s3://{}/{}'.format(bucket, key['Key']) for page in page_iterator for key in page['Contents'] if key['Key'].endswith(suffix)]
    return paths

def well_series_test(cluster, input_adata):
    contam_cells = input_adata.obs[input_adata.obs.louvain == cluster]
    return_df = pd.DataFrame()
    return_df['well'] = [x.split('_')[0] for x in contam_cells.index]
    return_df['plate'] = [x.split('_')[1] for x in contam_cells.index]
    
    for plate in set(return_df.plate):
        display(return_df[return_df.plate == plate].sort_values('well'))
        
def fast_DE(input_adata, clusterOI, groupby, reference='rest', n_genes=10):
    sc.tl.rank_genes_groups(input_adata, 
                            groupby=groupby, 
                            groups=[clusterOI], 
                            method='wilcoxon',
                            reference=reference,
                            n_genes = n_genes)
    gene = [x[0] for x in input_adata.uns['rank_genes_groups']['names']]
    return gene
# scaled heatmap to zero mean and unit variance
from sklearn.preprocessing import scale, MinMaxScaler
def min_max_scaler(x):
    scaler = MinMaxScaler()
    scaler.fit(x.reshape((-1,1)))
    return scaler.transform(x.reshape((-1,1))).reshape((-1,))

def value2key(value, dictionary):
    return_val = 'none'
    for key,values in dictionary.items():
        if value in values:
            return_val = key
            
    return return_val

from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
def rect_converter(df, xval, yval, y_upper, y_lower, grouping):
    master = pd.DataFrame()
    for label in set(df[grouping]):
        reassign = pd.DataFrame()
        df_slice = df[df[grouping] == label]
        reassign['xmin'] = df_slice[xval].shift(-1).tolist()
        reassign['xmax'] = df_slice[xval].tolist()
        reassign['ymin'] = df_slice[y_lower].tolist()
        reassign['ymax'] = df_slice[y_upper].tolist()
        reassign['label'] = label
        master = master.append(reassign[:-1])
    return master.dropna()

def regress(x, y, predictor, response, fit_intercept = False):
    # requires 1D array of shape (-1, 1)
    # returns fit and residuals residuals and slope
    
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

def calc_adj_pval(pval_list, n_hyp):
    # ref: https://www.researchgate.net/post/Can_anyone_explain_how_to_calculate_adjusted_p-values_q-values_following_Benjamini_Hochberg_correction
    pval_df = pd.DataFrame({'pval':pval_list})
    pval_df['rank'] = pval_df['pval'].rank(method='dense')
    pval_df['adj_pval_bh'] = [(x*n_hyp)/y for x,y in zip(pval_df['pval'],pval_df['rank'])]

    return pval_df['adj_pval_bh'].values.tolist()

def adata_DE_pairwise(input_adata, 
                      groupby, 
                      target_1, 
                      target_2, 
                      method = 'wilcoxon',
                      corr_method = 'benjamini-hochberg'
                     ):
    """This is a two-sided test!"""
    
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

                      