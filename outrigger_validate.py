#!/home/ubuntu/anaconda3/bin//python3 python3

import pandas as pd
import numpy as np
from subprocess import run
import os, sys, time, subprocess
from shutil import copyfile,rmtree

def pull_job(jobs_path):
    s3path = None
    
    # get instance id
    proc = run(['ec2metadata', '--instance-id'], 
                              encoding='utf-8', 
                              stdout=subprocess.PIPE)
    ec2_id =  proc.stdout.split('\n')[0]
    
    # pull jobs queue
    jobs_df = pd.read_csv(jobs_path)
    if ec2_id in jobs_df.ec2_id.values:
        s3path = jobs_df[jobs_df.ec2_id == ec2_id].path.tolist()[0]
    elif len(jobs_df) > 0:
        print('No matching jobs')
    else:
        print('No jobs in queue')
    return s3path

def module1(s3path):
    file_prefix = s3path.split('.')[0].split('/')[-1]
    prefix = '_'.join(file_prefix.split('_')[:2])
    plate = file_prefix.split('_')[1]

    return file_prefix, prefix, plate

def module2(s3path, wkdir):
    os.chdir('/home/ubuntu/')
    process = run(['aws', 's3', 'cp', s3path, f'{wkdir}/'])
    return process.returncode
    
def module3A(wkdir, file_prefix, gtf_file):
    os.chdir(wkdir)
    process = run(['outrigger', 'index', 
                 '--sj-out-tab', f'{file_prefix}.homo.SJ.out.tab',
                 '--gtf', gtf_file])
    return process.returncode

def module3B(wkdir, chrlen_file, fa_file):
    os.chdir(wkdir)
    process = run(['outrigger', 'validate', 
                 '--genome', chrlen_file,
                 '--fasta', fa_file])
    return process.returncode
    
def module4(wkdir, subtype, dest):
    os.chdir('/home/ubuntu/')
    process = run(['aws', 's3', 'cp',
                 f'{wkdir}/outrigger_output/index/{subtype}/validated/events.csv', 
                 f'{dest}/'])
    return process.returncode

def logging(wkdir, name, exit_code):
    with open(f'{wkdir}/log.txt', 'a') as f:
        f.write(f'{name}, {exit_code}\n')

def main(jobs_path):
    print(jobs_path)
    
    # pull jobs
    try:
        s3path = pull_job(jobs_path)
        
        # variables
        start_time = time.time()
        wkdir = f'/home/ubuntu/wkdir'
        gtf_file = '/home/ubuntu/data/HG38-PLUS/genes/genes.gtf'
        fa_file = '/home/ubuntu/data/HG38-PLUS/fasta/genome.fa'
        chrlen_file = '/home/ubuntu/data/HG38-PLUS/star/chrNameLength.txt'
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

        # run outrigger index and valide modules
        exit_code = module3A(wkdir, file_prefix, gtf_file)
        logging(wkdir, 'run_outrigger', exit_code)

        exit_code = module3B(wkdir, chrlen_file, fa_file)
        logging(wkdir, 'run_validate', exit_code)

        # compile results
        for subtype in ['se','mxe']:
            exit_code = module4(wkdir, subtype, dest)
            logging(wkdir, f'{subtype}_upload', exit_code)

        # record execution time
        try:
            etime = time.time() - start_time
        except:
            etime = -1
        logging(wkdir, '__exec_time', etime)
        
    except:
        print('abort process')
    
main(sys.argv[1])