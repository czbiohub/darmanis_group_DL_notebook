#!/home/ubuntu/anaconda3/bin//python3 python3

import pandas as pd
import numpy as np
import os, sys, time, subprocess
from shutil import copyfile,rmtree

def main():
    proc = subprocess.run(['ec2metadata', '--instance-id'], 
                          encoding='utf-8', 
                          stdout=subprocess.PIPE)
    ec2_id =  proc.stdout
    print(ec2_id)
    
main()