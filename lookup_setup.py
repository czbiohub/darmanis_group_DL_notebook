# base
import pandas as pd
from pandas.api.types import CategoricalDtype, is_categorical_dtype
import numpy as np
from scipy import sparse, stats
from scipy.stats import ttest_ind, ranksums, variation
import warnings
import scipy.stats as ss
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn import preprocessing
from sklearn.metrics import f1_score, roc_auc_score, jaccard_similarity_score
from IPython.core.display import HTML
from IPython.display import clear_output
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
#     sc.pp.regress_out(tmp, ['n_counts'])
#     print('REGRESSION ON')
    
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

def umap_adata (adata, res=None):
    # Perform clustering and UMAP
    # Input: adata obj + cluster options (below)
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
    
    # plot cluster occupancies
    res_vals = df.columns.tolist()
    base_df = (df.groupby(df.columns.tolist()[0]).size()
                    .reset_index()
                    .rename({0:'count',
                             df.columns.tolist()[0]:'group_2'}, axis='columns'))
    base_df['group_1'] = base_df['group_2']
    base_df['step'] = res_vals[0]
    base_df['ncounts'] = base_df['count']/base_df['count'].sum()
    
    for idx,x in enumerate(range(2,len(df.columns.tolist())+1,1)):
        columns_oi = df.columns.tolist()[(x-2):x]
        tmp = (df.groupby(columns_oi)
                .size()
                .reset_index()
                .rename({0:'count', 
                         columns_oi[0]:'group_1',
                         columns_oi[1]:'group_2'}, axis='columns'))
        tmp['step'] = res_vals[idx+1]
        tmp['ncounts'] = tmp['count']/tmp['count'].sum()
        base_df = base_df.append(tmp)

    groupcat1 = CategoricalDtype(['{}'.format(x) for x in range(len(set(base_df['group_1'])))],ordered=True)
    base_df['group_1_cat'] = base_df['group_1'].astype(str).astype(groupcat1)
    groupcat2 = CategoricalDtype(['{}'.format(x) for x in range(len(set(base_df['group_2'])))],ordered=True)
    base_df['group_2_cat'] = base_df['group_2'].astype(str).astype(groupcat2)

    plotnine.options.figure_size = (8,8)
    print(ggplot(base_df)
            +theme_bw()
            +theme(aspect_ratio=1)
            +geom_bar(aes('group_2_cat','ncounts',fill='group_1_cat'),stat='identity')
            +facet_wrap('~step')
            +labs(x='current label', y='proportion of cells', fill='previous label'))
    
    # plot max Jaccard index for each pair of resolutions
    # convert to int for addition operation
    for col in df.columns.tolist():
        df[col] = df[col].astype(int)

    # return max jaccard index value over n+1 label iterator
    jidx_list = []
    for idx, col in enumerate(res_vals):
        if idx > 0:
            ref = df[res_vals[idx-1]].tolist()
            curr = df[res_vals[idx]].tolist()

            poss_labs = list(set(ref+curr))
            jidx_poss = [jaccard_similarity_score(ref, curr)]
            for idx, x in enumerate(range(len(poss_labs)-1)):
                reset_val = max(poss_labs)
                ref = [(x+1) if (x <= reset_val) else 0 for x in ref]
                jidx_poss.append(jaccard_similarity_score(ref, curr))
            jidx_list.append(max(jidx_poss))
    jidx_df = pd.DataFrame({'jidx':jidx_list, 'res':res_vals[1:]})
    
    plotnine.options.figure_size = (3,3)
    print(ggplot(jidx_df)
             +theme_bw()
             +theme(aspect_ratio=1,
                    axis_text_x=element_text(angle=90))
             +geom_bar(aes('res','jidx'), stat='identity')
             +labs(y='Jaccard index (rel. to prev. res.)',x='current res.'))
    
def classify_type(raw_adata, clustered_adata, input_class, type_dict, output_class):
    # Manually classify
    # Input: raw ad obj (unfiltered) + ad obj with cluster assignments + dict of labels:cluster assignment + colname
    # Output: update raw ad obj in place
    
    type_list = ['unknown'] * len(raw_adata.obs)
    
    for key,value in type_dict.items():
        clustered_names = [name for cluster,name in zip(clustered_adata.obs[input_class], 
                                                             clustered_adata.obs_names) if cluster in value]  
        value_idx = [idx for idx,x in enumerate(raw_adata.obs_names) if x in clustered_names]
        for x in value_idx:
            type_list[x] = key
            
    raw_adata.obs[output_class] = type_list
    
def rank_genes (input_adata, methods=['wilcoxon','t-test_overestim_var'],n_genes=20, groupby='louvain'):
    # Rank genes
    # Input: ad obj
    # Output: dataframe of ranked genes

    stack_df = pd.DataFrame()
    for method in methods:
        print(method)
        sc.tl.rank_genes_groups(input_adata, groupby=groupby, method=method, n_genes=n_genes)
        df_rank = pd.DataFrame(input_adata.uns['rank_genes_groups']['names'])
        print(df_rank.head(10).to_string(index=False))
        out_df = pd.DataFrame()
        for x in tqdm.tqdm(df_rank.columns):
            genelist=df_rank.loc[:,str(x)].tolist()
            output=['None' for x in genelist]
            output=[lookup_gene(x, u) for x in genelist]
            funct_df=pd.DataFrame({'gene':genelist, 'function':[x[0] for x in output], 'GO':[x[1] for x in output]})
            funct_df[groupby] = str(x)
            out_df = out_df.append(funct_df)
        out_df['method'] = method
        stack_df = stack_df.append(out_df)
    
    return stack_df

def push_rank (df_rank, title, wkdir, s3dir):
    # save CSV of gene list to output to S3
    # Input: df of ranks + local/s3 paths + title
    # Output: push to s3
    
    rank_fn = 'GeneRank_{}.csv'.format(title)
    rank_path = '{}/{}'.format(wkdir, rank_fn)
    df_rank.to_csv(rank_path)
    s3_cmd = 'aws s3 cp --quiet {}/{} s3://{}/'.format(wkdir,rank_fn,s3dir)
    subprocess.run(s3_cmd.split()) # push to s3
    subprocess.run(['rm', rank_path]) # remove local copy    
    
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

def occupancy(input_adata, elements, group):
    # occupancy of group
    # input: str element name, str group name, adata obj
    # output: plots showing occupancy


    patients = input_adata.obs[elements].tolist()
    clusters = input_adata.obs[group].tolist()

    # determine counts per cluster by patient class
    df = pd.DataFrame({'cluster':clusters, elements:patients})
    df = df.groupby(['cluster',elements]).size()
    df = df.unstack().replace(np.nan, 0)
    df = df.reset_index()
    # print(df.to_string(index=False))
    df = pd.melt(df, id_vars='cluster')

    plotnine.options.figure_size = (4,4)
    print(ggplot(df, aes('cluster','value',fill=elements))+
         theme_bw()+
         theme(aspect_ratio=1)+
         geom_bar(stat='identity')+
         labs(y='# of cells',x=''))

    # normalize by cluster
    df['value'] = df.groupby('cluster').transform(lambda x: x/x.sum())
    plotnine.options.figure_size = (4,4)
    print(ggplot(df, aes('cluster','value',fill=elements))+
         theme_bw()+
         theme(aspect_ratio=1)+
         geom_bar(stat='identity')+
         labs(y='fraction of elements',x=''))


def main():
    # download data from s3
    wkdir = '/home/ubuntu/data/DL20181011_melanocyte_test_data'
    s3dir = 'daniel.le-work/MEL_project'

    # read annotation data
    anno = pd.read_csv('{}/primary_mel_metadata_181011.csv'.format(wkdir), index_col=0)
    anno = anno.loc[:, ['nGenes', 'nReads', 'well', 'plate', 'patient_id']]
    anno = anno.rename(columns={'plate':'plate_barcode'})
    anno['cell_name'] = ['{}_{}'.format(x,y) for x,y in zip(anno['well'], anno['plate_barcode'])]

    # metadata update: 11-07-2018
    plate_df = pd.read_csv('{}/DL20181107_metadata_update.csv'.format(wkdir))
    plate_df = plate_df.loc[~plate_df['plate_barcode'].isnull(),:] # remove any row without plate barcode
    plate_df = plate_df.loc[:, ['plate_barcode', 'patient_id', 'sample_color', 'age', 
                                'age_bin', 'sex','race', 'general_location', 'anatomical_location']]

    # update metadata with merge
    merged_anno = pd.merge(anno, plate_df, 'left', ['plate_barcode', 'patient_id'])


    # append bsc metadata to anno
    bsc = pd.read_csv('{}/DL20181106_bsc_metadata.csv'.format(wkdir))
    merged_anno = pd.merge(merged_anno,bsc,'left',['well','plate_barcode'])
    merged_anno = merged_anno.set_index('cell_name')

    # read raw data (expression data) to pandas df
    pre_adata = pd.read_csv('{}/primary_mel_rawdata_181011.csv'.format(wkdir), index_col=0)
    pre_adata.columns = merged_anno.index.tolist()
    pre_adata = pre_adata.rename_axis('gene_name')

    # markers
    markers = ['PMEL','KRT1','KRT5','KRT10','TYR','MITF']

    # ingest data
    raw_adata = create_adata(pre_adata)
    raw_adata.var['ribo'] = raw_adata.var_names.str.startswith(('RPL','RPS'))
    raw_adata.var['ercc'] = np.array([True if 'ERCC' in x else False for x in raw_adata.var_names.tolist()])
    sc.pp.calculate_qc_metrics(raw_adata, feature_controls=['ribo','ercc'], inplace=True)
    anno_dict = {'age':'age',
                 'age_bin':'age_bin',
                 'plate':'plate_barcode',
                 'general_location':'general_location',
                 'anatomical_location':'anatomical_location',
                 'race':'race',
                 'sex':'sex',
                 'color':'sample_color',
                 'patient':'patient_id',
                 'bsc': 'bsc_a'}
    append_anno(raw_adata, merged_anno, anno_dict)
    append_markers(raw_adata, gene_markers=markers)
    technical_filters(raw_adata)
    # raw_adata = remove_ercc(raw_adata) 
    raw_adata.raw = sc.pp.log1p(raw_adata, copy=True) # freeze raw state

    # all-cells analysis
    full_adata = process_adata(raw_adata)
    pca_adata(full_adata, num_pcs=16)
    umap_adata(full_adata, res=0.3)

    # classify in raw adata
    input_adata = full_adata

    MEL_int = [0,1,2,4]
    KRT_int = [x for x in range(len(input_adata.obs['louvain'].cat.categories)) if x not in MEL_int]
    type_dict = {'KRT':[str(x) for x in KRT_int],
                 'MEL':[str(x) for x in MEL_int]}

    classify_type(raw_adata, input_adata, 'louvain', type_dict, 'class_1')

    # remove non-Adult bins
    age_bins = raw_adata.obs['age_bin'].unique()
    for feat in ['FET_12WK','NEO']:
        age_bins = age_bins[age_bins != feat]

    feature_dict = {'age_bin':age_bins.tolist(), 'class_1':['MEL']}
    adata_subset1 = subset_adata_v3(raw_adata,feature_dict)
    adata_subset1 = process_adata(adata_subset1)
    pca_adata(adata_subset1, num_pcs=7)
    umap_adata(adata_subset1, res=0.1)

    # cull cells using Tukey outlier threshold
    df = pd.DataFrame(adata_subset1.obsm['X_pca'])
    df.columns = ['PC_{}'.format(x) for x in range(50)]

    omit_names = []
    for pc in df.columns:
    #     print(len(set(omit_names)))
        Q1 = df[pc].quantile(0.25)
        Q3 = df[pc].quantile(0.75)
        IQR = Q3 - Q1
        threshold=7

        bool_idx = [True if ((Q1 - threshold * IQR) <= x <= (Q3 + threshold * IQR)) else False for x in df[pc]]
        filtered = df[bool_idx]
    #     print('Filtered cells: ', len(df)-len(filtered))

        omit_names = omit_names + adata_subset1.obs[[not x for x in bool_idx]].index.tolist()

    print('Filtered cells: ', len(set(omit_names)))
    test = raw_adata[[x for x in raw_adata.obs.index if x not in set(omit_names)]]

    # check where the outliers lie in original projection
    input_adata = adata_subset1
    input_adata.obs['outliers'] = ['outlier' if x in set(omit_names) else 'ingroup' for x in input_adata.obs.index]

    # outliers on all-cells projection
    input_adata = adata_subset1

    type_dict = {'ingroup':['ingroup'],
                 'outlier':['outlier']}

    classify_type(full_adata, input_adata, 'outliers', type_dict, 'outliers')

    # classify
    input_adata = adata_subset1

    type_dict = {'ingroup':['ingroup'],
                 'outlier':['outlier']}

    classify_type(raw_adata, input_adata, 'outliers', type_dict, 'class_2')

    # recluster
    feature_dict = {'age_bin':age_bins.tolist(), 'class_1':['MEL'], 'class_2':['ingroup']}
    adata_subset2 = subset_adata_v3(raw_adata, feature_dict)
    adata_subset2 = process_adata(adata_subset2)
    pca_adata(adata_subset2, num_pcs=30)
    umap_adata(adata_subset2, res=.5)

    # classify
    input_adata = adata_subset2

    type_dict = {'main':['0','1'],
                 'outer':['2','3','4']}

    classify_type(raw_adata, input_adata, 'louvain', type_dict, 'class_3')

    # recluster
    feature_dict = {'age_bin':age_bins.tolist(), 
                    'class_1':['MEL'], 
                    'class_3':['main']}
    adata_subset3 = subset_adata_v3(raw_adata, feature_dict)
    adata_subset3 = process_adata(adata_subset3)
    pca_adata(adata_subset3, num_pcs=30)
    umap_adata(adata_subset3, res=.4)
    clear_output()
    
    return full_adata, adata_subset1, adata_subset2, adata_subset3

def multi_gene (gene_list, adatas):
    warnings.filterwarnings('ignore')
    desc = ["all-cells", "adt-mel", "adt-mel sans outlier cells", "adt_mel sans all outliers"]
    desc_key = dict(zip([str(x) for x in adatas], desc))

    for adata in adatas:
        print(desc_key[str(adata)])
        sc.pl.umap(adata, color=gene_list + ["louvain"])
        
        input_adata = adata
        groupby='louvain'

        cats, casted_df = prepare_dataframe(input_adata, 
                                             gene_list, 
                                             groupby=groupby)

        melt_df = pd.melt(casted_df.reset_index(), id_vars=groupby)

        plotnine.options.figure_size = (8,8)
        print(ggplot(melt_df, aes(groupby,'value',color=groupby))
              +theme_bw()
              +theme(aspect_ratio=1)
              +coord_flip()
              +geom_boxplot()
              +facet_wrap('~variable', nrow=len(gene_list)//3))

        print(ggplot(melt_df, aes('value',fill=groupby))
              +theme_bw()
              +theme(aspect_ratio=1)
              +stat_bin(aes(y='stat(ncount)'))
              +facet_wrap('~variable', nrow=len(gene_list)//3))
        
def single_gene(gene, adatas):
    warnings.filterwarnings('ignore')
    desc = ["all-cells", "adt-mel", "adt-mel sans outlier cells", "adt_mel sans all outliers"]
    desc_key = dict(zip([str(x) for x in adatas], desc))
    for adata in adatas:
        print(desc_key[str(adata)])
        sc.pl.umap(adata, color=[gene, "louvain"])
        sc.pl.violin(adata, keys=gene, groupby="louvain")
        