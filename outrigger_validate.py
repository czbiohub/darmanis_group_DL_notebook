#!/home/ubuntu/anaconda3/bin//python3 python3

import pandas as pd
import numpy as np
from subprocess import run
import os, sys, time
from shutil import copyfile,rmtree

def module1(s3path):
    file_prefix = s3path.split('.')[0].split('/')[-1]
    prefix = '_'.join(file_prefix.split('_')[:2])
    plate = file_prefix.split('_')[1]

    return file_prefix, prefix, plate

def module2(s3path, wkdir):
    os.chdir('/home/ubuntu/')
    return_code = run(['aws', 's3', 'cp', s3path, f'{wkdir}/'])
    return return_code
    
def module3(wkdir, file_prefix, gtf_file, fa_file):
    os.chdir(wkdir)
    return_code1 = run(['outrigger', 'index', 
                 '--sj-out-tab', f'{file_prefix}.homo.SJ.out.tab',
                 '--gtf', gtf_file])
    os.chdir(wkdir)
    return_code2 = run(['outrigger', 'validate', 
                 '--genome', 'hg38',
                 '--fasta', fa_file])
    return return_code1, return_code2
    
def module4(wkdir, subtype, dest):
    os.chdir('/home/ubuntu/')
    return_code = run(['aws', 's3', 'cp',
                 f'{wkdir}/outrigger_output/index/{subtype}/validated/events.csv', 
                 f'{dest}/'])
    return return_code

def logging(wkdir, name, exit_code):
    with open(f'{wkdir}/log.txt', 'a') as f:
        f.write(f'{name}, {exit_code}\n')

def main(s3path):
    print(s3path)
    
    # variables
    start_time = time.time()
    wkdir = f'/home/ubuntu/wkdir'
    gtf_file = '/home/ubuntu/data/HG38-PLUS/genes/genes.gtf'
    fa_file = '/home/ubuntu/data/HG38-PLUS/fasta/genome.fa'
    dest = 's3://daniel.le-work/MEL_project/DL20190111_outrigger'
    
    # parse path for prefix to name outputs
    try:
        file_prefix, prefix, plate = module1(s3path)
        exit_code = 0
    except:
        exit_code = 1
    logging(wkdir, 'parse_path', exit_code)
    
    # pull input from s3
    exit_code = module2(s3path, wkdir)
    logging(wkdir, 's3_download', exit_code)
    
#     # run outrigger and validate modules
#     exit_code = module3(wkdir, file_prefix, gtf_file, fa_file)
#     logging(wkdir, 'run_analysis', exit_code)

#     # compile results
#     for subtype in ['se','mxe']:
#         exit_code = module4(wkdir, subtype, dest)
#         logging(wkdir, f'{subtype}_upload', exit_code)
                     
#     # record execution time
#     try:
#         etime = time.time() - start_time
#     except:
#         etime = -1
#     logging(wkdir, '__exec_time', etime)
    
main(sys.argv[0])