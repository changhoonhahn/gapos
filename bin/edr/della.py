import os, sys 
import numpy as np 


def pSMF(targ, zmin, zmax): 
    ''' deploy pSMF fitting script 
    '''
    # next, write slurm file for submitting the job
    a = '\n'.join([
        '#!/bin/bash',
        '#SBATCH -J psmf.%s.z%.1f_%.1f' % (targ, zmin, zmax), 
        "#SBATCH --mem=8G", 
        '#SBATCH --time=02:00:00',
        "#SBATCH --export=ALL",
        '#SBATCH -o o/psmf.%s.z%.1f_%.1f' % (targ, zmin, zmax), 
        "#SBATCH --mail-type=all",
        "#SBATCH --mail-user=chhahn@princeton.edu",
        "#SBATCH --gres=gpu:1" 
        '',
        "module load anaconda3/2021.11", 
        "conda activate sbi", 
        '', 
        'python /home/chhahn/projects/proSMF/bin/edr/psmf.py %s %.1f %.1f' % (targ, zmin, zmax), 
        ''])
        
    # create the script.sh file, execute it and remove it
    f = open('script.slurm','w')
    f.write(a)
    f.close()
    os.system('sbatch script.slurm')
    os.system('rm script.slurm')
    return None 


def pSMF_jack(targ, zmin, zmax): 
    ''' deploy pSMF fitting script 
    '''
    # next, write slurm file for submitting the job
    a = '\n'.join([
        '#!/bin/bash',
        '#SBATCH -J psmf.%s.z%.1f_%.1f' % (targ, zmin, zmax), 
        "#SBATCH --mem=8G", 
        '#SBATCH --time=02:00:00',
        "#SBATCH --export=ALL",
        '#SBATCH -o o/psmf.%s.z%.1f_%.1f' % (targ, zmin, zmax), 
        "#SBATCH --mail-type=all",
        "#SBATCH --mail-user=chhahn@princeton.edu",
        "#SBATCH --gres=gpu:1" 
        '',
        "module load anaconda3/2021.11", 
        "conda activate sbi", 
        '', 
        'python /home/chhahn/projects/proSMF/bin/edr/psmf_jack.py %s %.1f %.1f' % (targ, zmin, zmax), 
        ''])
        
    # create the script.sh file, execute it and remove it
    f = open('script.slurm','w')
    f.write(a)
    f.close()
    os.system('sbatch script.slurm')
    os.system('rm script.slurm')
    return None 



zmins = [0.01 + 0.04 * i for i in range(8)]
zmaxs = [0.01 + 0.04 * (i+1) for i in range(8)]

for zmin, zmax in zip(zmins, zmaxs): 
    #pSMF('bgs_bright', zmin, zmax)
    pSMF('bgs_any', zmin, zmax) 
    pSMF_jack('bgs_any', zmin, zmax) 
