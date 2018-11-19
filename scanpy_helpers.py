# base
import pandas as pd
from pandas.api.types import CategoricalDtype, is_categorical_dtype
import numpy as np
from scipy import sparse, stats
import warnings
import scipy.stats as ss
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn import preprocessing
from sklearn.metrics import f1_score, roc_auc_score
from IPython.core.display import HTML
import numpy.ma as ma # masking package
import statsmodels.api as sm
import random
import subprocess
import pickle
import tqdm
from itertools import combinations, permutations

# sc analysis
import scanpy.api as sc
import anndata as ad
import bbknn
import gseapy as gp

#plotting
from plotnine import *
import plotnine
import matplotlib as mp

# uniprot api
from bioservices import UniProt
u = UniProt()

### ref scanpy docs
def prepare_dataframe(adata, var_names, groupby=None, use_raw=None, log=False, num_categories=7):
    """
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
###



def gene2exp (gene_str, adata):
    # Create array of transformed expression values from given gene symbol string
    # Input: str gene name + ad obj with expression embedded
    # Outupt: log10 expression arrays that can be appended to adata obj
    
    gene_lin = adata[:, gene_str].X.tolist()
    gene_log10 = np.log10(gene_lin)
    gene_log10[np.isinf(gene_log10)] = 0
    
    return gene_log10

def append_markers (adata, gene_markers):
    # Appends gene expression as annotation data in order to plot
    # Input: adata obj + gene list
    # Output: updated adata obj
    
    print('Append marker gene expresssion...')
    
    try_markers = gene_markers
    for gene_name in try_markers:
        try:
            adata.obs[gene_name] = gene2exp(gene_name, adata)
        except:
            pass

def sum_output (adata):
    # Prints cell and gene count
    # Input: ad obj
    # Output: print out
    print('\tCells: {}, Genes: {}'.format(len(adata.obs), len(adata.var_names)))

def create_adata (pre_adata):
    # Creates adata obj from raw data (rows=gene_names, col=cell_id)
    # Input: raw expression data in pd df
    # Output: adata obj
    
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
    
    # summary
    sum_output (adata)
    
    return adata
    
def append_anno (adata, anno, anno_dict):
    # Add annotations of choice from annotation file
    # input = adata obj + dictionary of label and column name (with respect to annotation df) + anno pd df
    # output = updated adata obj
    
    print('Append annotations...')
    
    anno = anno
    anno_dict = anno_dict
    
    # append metadata of choice
    for key,value in anno_dict.items():
        adata.obs[key] = eval('anno.{}.values'.format(value))
    
    # summary
    sum_output (adata)

def remove_ercc (adata):
    # Remove ercc spike-in
    # Input: adata obj
    # Output: updated adata obj
    
    print('Remove ERCC genes...')
    
    gene_names = adata.var_names.tolist()
    ERCC_hits = list(filter(lambda x: 'ERCC' in x, gene_names))
    adata = adata[:, [x for x in gene_names if not (x in ERCC_hits)]]
    
    # summary
    print('Filtered genes: {}'.format(len(ERCC_hits)))    
    sum_output (adata)
    
    return adata

def technical_filters (adata, min_genes=500,min_counts=50000,min_cells=3):
    # remove cells/genes based on low quality
    # input: adata
    # output: inplace
    print('Remove low-quality cells/genes...')
    print('\tInitial:')
    sum_output (adata)
    
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_cells(adata, min_counts=min_counts)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    
    print('\tResult:')   
    sum_output (adata)

def process_adata (adata, 
                  min_mean=0.0125, 
                  max_mean=10, 
                  min_disp=0.1):
    # Add cell and gene filters, perform data scale/transform
    # Input: adata obj + filter options (below)
    # Output: updated adata obj
    
    print('Process expression data...')
    print('\tInitial:')
    sum_output(adata)

    print('min mean expression (min_mean): {}'.format(min_mean))
    print('max mean expression (max_mean)): {}'.format(max_mean))
    print('min dispresion (min_disp): {}'.format(min_disp))
    
    # normalize counts per cell
    tmp = sc.pp.normalize_per_cell(adata, copy=True)
    
    # filter cells based on min genes and min counts cutoff
    filter_result = sc.pp.filter_genes_dispersion(tmp.X, 
                                                  min_mean=min_mean, 
                                                  max_mean=max_mean, 
                                                  min_disp=min_disp)
    tmp = tmp[:, filter_result.gene_subset]
    
    # log transform expression
    sc.pp.log1p(tmp)
    
    # regress out total counts
    sc.pp.regress_out(tmp, ['total_counts'])
    
    # mean-center and unit variance scaling
    sc.pp.scale(tmp)
    
    # summary
    print('Filtered cells: {}'.format(len(adata.obs) - len(tmp.obs)))
    print('Filtered genes: {}'.format(len(adata.var_names) - len(tmp.var_names)))
    print('\tResult:')
    sum_output (tmp)
    
    return tmp

def pca_adata (adata, num_pcs=None, hoods=30):
    # Perform PCA dimensionality reduction + plot variance explained
    # Input: adata obj
    # Output: updated adata obj + print plot
    
    print('Principle component analysis...')
    
    # Perform PCA
    sc.tl.pca(adata)
    adata.obsm['X_pca'] *= -1  # multiply by -1 to match Seurat
    sc.pl.pca_variance_ratio(adata, log=True)
    
    # Neigbhor graph
    if num_pcs is None:
        print('Enter number of principle components to use:')
        input1=input()
        try:
            num_pcs = int(input1)
        except Exception as e:
            print(e)
            print('Using default settings')
            num_pcs=15
    
    print('principle_components(num_pcs): {}\ncells/neighborhood(hoods): {}'.format(num_pcs, hoods))
    sc.pp.neighbors(adata,n_pcs=num_pcs, n_neighbors=hoods)

def umap_adata (adata, meta_labels=None, res=None):
    # Perform clustering and UMAP
    # Input: adata obj + cluster options (below)
        # num_pcs = number of principle components
        # hoods = number of cells in neighborhood
        # res = resolution of community detection
    # Output: updated adata + UMAP to std out
    
    print('Uniform manifold approximation and projection...')
    
    # sample resolutions for louvain clustering
    print('\tScan Louvain detection resolutions')
    scan_res(adata) 
    
    if res is None:
        print('Enter Louvain detection resolution to use:')
        input1=input()
        try:
            res=float(input1)
        except Exception as e:
            print(e)
            print('Using default settings')
            res=0.5
    
    print('resolution(res): {}'.format(res))
    
    # UAMP: Uniform Maniford Approximation and Projection
    sc.tl.umap(adata)

    # Louvain community clustering
    sc.tl.louvain(adata, resolution = res)
    sc.pl.umap(adata, color=['louvain'], legend_loc='on data')
    
    # Plot metadata
    if meta_labels is None:
        meta_labels = sorted([x for x in adata.obs.columns.values.tolist() if (x is not 'louvain')])
        sc.pl.umap(adata, color=meta_labels)
    else:
        pass
    
def subset_adata (adata, feature_dict):
    # Subset adata obj by user dictionary of features and values list. Filter operation below:
        # level1: value1 | value2 | value3...
        # level2: key1 & key2 & key3...
    # Input: raw_adata obj + dictionary of feature:[value1, value2...]
    # Output: subsetted adata obj
    
    input_adata = adata
    stack_len = len(input_adata.obs)
    subset_arr = np.zeros((len(feature_dict), stack_len))
    
    print('Subsetting data...')
    
    key_count = 0
    for key,value in feature_dict.items(): 
        depth = 0
        stack_depth = len(value)
        stack = np.empty((stack_depth, stack_len))
        for val in value:
            stack[depth, :] = np.array(input_adata.obs[key] == val, dtype='bool')
            print('key = {}, value = {}, matched = {}'.format(key,val,np.sum(stack[depth, :], dtype='int')))
            depth += 1    
        subset_value = np.sum(stack, axis = 0, dtype='bool')
        subset_arr[key_count, :] = subset_value
        key_count += 1
        
    subset_list = np.product(subset_arr, axis = 0, dtype='bool')    
    input_adata = input_adata[subset_list,:]
    sum_output (input_adata)
    
    return input_adata

def subset_adata_v2 (raw, subset, feature_dict):
    # Subset adata obj by user dictionary of features and values list. Filter operation below:
        # level1: value1 | value2 | value3...
        # level2: key1 & key2 & key3...
    # Input: raw_adata obj + subsetted ad obj + dictionary of feature:[value1, value2...]
    # Output: subsetted adata obj
    
    input_adata = subset
    stack_len = len(input_adata.obs)
    subset_arr = np.zeros((len(feature_dict), stack_len))
    
    print('Subsetting data...')
    
    # subset the subset
    key_count = 0
    for key,value in feature_dict.items(): 
        depth = 0
        stack_depth = len(value)
        stack = np.empty((stack_depth, stack_len))
        for val in value:
            stack[depth, :] = np.array(input_adata.obs[key] == val, dtype='bool')
            print('key = {}, value = {}, matched = {}'.format(key,val,np.sum(stack[depth, :], dtype='int')))
            depth += 1    
        subset_value = np.sum(stack, axis = 0, dtype='bool')
        subset_arr[key_count, :] = subset_value
        key_count += 1
        
    subset_list = np.product(subset_arr, axis = 0, dtype='bool')    
    input_adata = input_adata[subset_list,:]
    
    # match cell names with raw
    matches = [True if x in input_adata.obs_names.tolist() else False for x in raw.obs_names.tolist()]
    output_adata = raw[matches,:] 
    sum_output (output_adata)
    
    return output_adata

def subset_adata_v3 (raw, feature_dict):
    stack_len = len(raw)
    bool_arr = np.zeros((len(feature_dict), stack_len))
    key_count = 0
    for key,value in feature_dict.items(): 
        depth = 0
        stack_depth = len(value)
        stack = np.empty((stack_depth, stack_len))
        for val in value:
            stack[depth, :] = np.array(raw.obs[key] == val, dtype='bool')
            print('key = {}, value = {}, matched = {}'.format(key,val,np.sum(stack[depth, :], dtype='int')))
            depth += 1    
        subset_value = np.sum(stack, axis = 0, dtype='bool')
        bool_arr[key_count, :] = subset_value
        key_count += 1
    subset_list = np.product(bool_arr, axis = 0, dtype='bool')    
    output_adata = raw[subset_list,:] 
    sum_output (output_adata)
    
    return output_adata    
    
def class2continuous_reg (X, y, test_size = 0.33):
    # Linear regression and returns R2
    # Input: list/array of predictors (categorical = str) and list of responses (float)
    # Output: R2 value
    
    pred = X # must be categorical
    res = y # must be continuous
    factor_levels = np.unique(pred)
    factor_len = len(factor_levels)

    df = pd.DataFrame({'pred':pred})
    df = pd.get_dummies(df)
    df['res'] = res

    X_train, X_test, y_train, y_test = train_test_split(df[df.columns[:factor_len]],df[df.columns[factor_len]],test_size=test_size, random_state=42)

    lm = LinearRegression()
    lm.fit(X=X_train, y=y_train)
    
    return lm.score(X=X_test, y=y_test)

def class2class_reg (X, y, test_size=0.33):
    # Logistic regression and returns accuracy
    # Input: list/array of predictors (categorical = str) and list of responses (categorical = str)
    # Output: accuracy value

    pred = X # must be categorical
    res = y # must be categorical
    factor_levels = np.unique(pred)
    factor_len = len(factor_levels)
    
    if len(np.unique(res)) == 1:
        acc = 0
    else:

        df = pd.DataFrame({'pred':pred})
        df = pd.get_dummies(df)
        df['res'] = res

        X_train, X_test, y_train, y_test = train_test_split(df[df.columns[:factor_len]],df[df.columns[factor_len]],test_size=test_size, random_state=42)

        # accurcy
        clf = LogisticRegression(multi_class='auto')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # try label-sized adjusted f1 score
        acc  = f1_score(y_pred=y_pred, y_true=y_test,average='micro')
    
    return acc

def scan_res(input_adata, step_size=0.05):
    # Scan through resolution setting for Louvain clustering and return similarity to previous step. Used to determine resolution setting.
    # Input: ad obj + step size (optional; float)
    # Output: print plot
    
    print('\tresolution_interval(step_size): {}'.format(step_size))
    
    basis_point = int(step_size * 100)
    
    # compute clusters
    df = pd.DataFrame()
    res_list = [x/100 for x in range(basis_point,100,basis_point)]
    for res in res_list:
        tmp = input_adata
        sc.tl.louvain(tmp, resolution = res)
        df['res_{}'.format(res)] = tmp.obs['louvain']
    
    # compute log reg
    col_labels = df.columns.tolist()
    acc_list = []
    for idx in range(len(col_labels)-1):
        acc_list.append(class2class_reg(X=df[col_labels[idx]].values, y=df[col_labels[idx+1]].values))

    sim_df = pd.DataFrame({'resolution':res_list[1:],
                          'similarity':acc_list})
    
    plotnine.options.figure_size = (2,2)

    print(ggplot(sim_df, aes(x='resolution',y='similarity'))+
        theme_bw()+
         theme(aspect_ratio=1)+
         geom_line())
    
def classify_type(raw_adata, clustered_adata, type_dict, col_name):
    # Manually classify MEL vs KRT
    # Input: raw ad obj (unfiltered) + ad obj with cluster assignments + dict of labels:cluster assignment + colname
    # Output: update raw ad obj in place
    
    type_list = ['unknown'] * len(raw_adata.obs)
    
    for key,value in type_dict.items():
        clustered_names = [name for cluster,name in zip(clustered_adata.obs['louvain'], 
                                                             clustered_adata.obs_names) if cluster in value]  
        value_idx = [idx for idx,x in enumerate(raw_adata.obs_names) if x in clustered_names]
        for x in value_idx:
            type_list[x] = key
            
    raw_adata.obs[col_name] = type_list
    
def rank_genes (input_adata, n_genes=100, method='wilcoxon'):
    # Rank genes
    # Input: ad obj
    # Output: dataframe of ranked genes
    
    sc.tl.rank_genes_groups(input_adata, groupby='louvain', method=method, n_genes=n_genes)
    df_rank = pd.DataFrame(input_adata.uns['rank_genes_groups']['names'])
    
    return df_rank

def push_rank (df_rank, feature_dict, wkdir, s3dir, method):
    # save CSV of gene list to output to S3
    # Input: df of ranks + subsetting feature dictionary + local/s3 paths + method string
    # Output: push to s3
    
    id_string = '_'.join(['{}.{}'.format(key,'-'.join(value)) for key,value in feature_dict.items()])
    rank_fn = 'GeneRank_{}_{}.csv'.format(method, id_string)
    rank_path = '{}/{}'.format(wkdir, rank_fn)
    df_rank.to_csv(rank_path)
    s3_cmd = 'aws s3 cp --quiet {}/{} s3://{}/'.format(wkdir,rank_fn,s3dir)
    subprocess.run(s3_cmd.split()) # push to s3
    #subprocess.run(['rm', rank_path]) # remove local copy    
    
    # print s3 download link
    dl_link = 'https://s3-us-west-2.amazonaws.com/{}/{}'.format(s3dir,rank_fn)
    print(dl_link)
    
def PC_contribution (target, input_adata):
    # Determine gene contribution to PC
    # Input: gene name + ad obj
    # Output: print plots
    
    loadings = input_adata[:,target].varm[0][0].tolist()
    max_loadings = [np.max(np.array([input_adata.varm[row][0][pc] for row in range(input_adata.varm.shape[0])])) for pc in range(50)]
    norm_loadings = [x/y for x,y in zip(loadings, max_loadings)]
    plot_df = pd.DataFrame({'norm_loadings':norm_loadings, 
                           'PCs':[x for x in range(50)]})

    sc.pl.umap(input_adata, color=[target])

    plotnine.options.figure_size = (5,5)
    print(ggplot(plot_df, aes('PCs', 'norm_loadings'))+
         theme_bw() +
         theme(aspect_ratio = 1) +
         geom_bar(stat='identity', position='dodge') +
         labs(y='relative fractional loading'))
    
def lookup_gene(symbol, u, warnings=False):
    # UniProt search
    # Input: gene symbol + iniprot interface obj
    # Output: annotation and GO term as tuple
    
    try:
        res = u.search(query="{}+AND+HUMAN".format(symbol),
                       columns= 'comment(FUNCTION), go(molecular function)', 
                       limit=1).split('\n')[1].split('\t')
    except Exception as e1:
        if warnings is True:
            print(symbol, e1)
        res = ['NA']*2
        

    if res[0] is '':
        res[0] = 'NA'
    if res[1] is '':
        res[1] = 'NA'
        
    annote = res[0]
    go = res[1]
    
    return annote, go
    
def continuous2class_reg (X, y, test_size=0.33):
    # Logistic regression and returns accuracy
    # Input: list/array of predictors (numeric) and list of responses (categorical = str)
    # Output: accuracy value

    pred = X # must be continuous
    res = y # must be categorical
    pred = pred.reshape(-1,1)
    res = res.reshape(-1,1)
    
    if len(np.unique(res)) == 1:
        acc = 0
    else:
        X_train, X_test, y_train, y_test = train_test_split(pred,
                                                            res,
                                                            test_size=test_size, random_state=42)
        # accurcy
        clf = LogisticRegression(multi_class='auto')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc  = f1_score(y_pred, y_test, average='micro')
    
    return acc
