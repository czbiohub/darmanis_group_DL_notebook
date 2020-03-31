import sys
import pysam
import numpy as np
import pandas as pd
from multiprocessing import Pool

inputfa = sys.argv[1]
inputbam = sys.argv[2]
outputfa = sys.argv[3]
ncore = int(sys.argv[4])

print(inputfa, inputbam, outputfa, ncore)

def find_segment(df):
    """
    Takes coverage df
    Finds all segments passing filter
    Takes intagral of segment reads
    Returns ref pos of largest integral
    """
    previous_element = None
    track_counter = 0
    track_dict = {}
    for idx, element in enumerate(df['passing']):

        if element == True:
            track_counter += 1
            if track_counter == 1:
                start = idx
        else:
            track_counter = 0
            if previous_element == True:
                end = idx
                df_slice = df.iloc[start:end, :]
                sum_piles = df_slice['cov'].sum()
                track_dict[sum_piles] = (df.iloc[start,:]['refpos'], df.iloc[end,:]['refpos'])
        
        previous_element = element
    if element == True:
        end = idx
        df_slice = df.iloc[start:end, :]
        sum_piles = df_slice['cov'].sum()
        track_dict[sum_piles] = (df.iloc[start,:]['refpos'], df.iloc[end,:]['refpos'])
    
    if len(track_dict.keys()) > 0:
        returnval = track_dict.get(max(list(track_dict.keys())))
    else:
        returnval = (0,0)
    
    return returnval

def trim_cluster (clust_id, frac=0.33, window=10, minlen=500):
    """
    Takes cluster id
    Pulls pileup information + computes segments
    returns trimmed seq
    """
    with pysam.FastaFile(inputfa) as fafile:
        clust_seq = fafile.fetch(clust_id).strip()
        cov_list=[]
        pos_list=[]
        with pysam.AlignmentFile(inputbam, "rb" ) as bamfile:
            for c in bamfile.pileup(reference=clust_id):
                pos_list.append(c.reference_pos)
                real_cov = 0
                for pileupread in c.pileups:
                    if not pileupread.is_del and not pileupread.is_refskip:
                        real_cov += 1
                cov_list.append(real_cov)

            # must have at least 1 read to keep cluster and len greater than minlen
            if any([x>0 for x in cov_list]) and len(pos_list) >= minlen:
                cov_df = pd.DataFrame({'cov':cov_list,'refpos':pos_list})
                cov_df['rollmean'] = cov_df['cov'].rolling(window).mean()
                # require > 10% of max rolling mean
                cov_df['passing'] = [x > (frac*np.max(cov_df['rollmean'])) for x in cov_df['rollmean']]
                segment = find_segment(cov_df)
                segment_start = segment[0] 
                segment_end = segment[1] 
                updated_clust_seq = clust_seq[segment_start:segment_end]
                if len(updated_clust_seq) < minlen:
                    updated_clust_seq = ''
            else:
                updated_clust_seq = ''
            
    return updated_clust_seq

def main():
    """
    Parallel fasta entry trim
    """
    with Pool(processes=ncore) as p:
        with pysam.FastaFile(inputfa) as fafile:
            updated_seq_list = p.map(trim_cluster, fafile.references)
            with open(outputfa, 'w') as outfile:
                for name,seq in zip(fafile.references, updated_seq_list):
                    if seq != '':
                        outfile.write('>' + name + '\n' + seq + '\n')

if __name__ == "__main__":
    main()