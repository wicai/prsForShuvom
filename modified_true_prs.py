import os
from config import *
import argparse

import numpy as np, pandas as pd, math
np.seterr(divide='ignore', invalid='ignore')
import msprime
import sys, threading
import gzip, h5py, os
from scipy import stats
import statsmodels.api as sm
import tqdm
import gzip
from scipy import stats
from scipy.stats import chi2

import matplotlib.pyplot as plt
import seaborn as sns
from simulation import *


m = M
h2 = H2
n_admix = 5000
outdir = 'output/'
sim = 1
prefix = f'{outdir}sim{sim}/'
p = P
r2 = R2
vcf_file = "admixed_data/output/admix_afr_amer.query.vcf.gz"
path_tree_CEU="trees/tree_CEU_GWAS_nofilt.hdf"
path_tree_YRI="trees/tree_YRI_GWAS_nofilt.hdf"
num2decrease=None
ld_distance=1e6
num_threads=16
snp_weighting = 'ceu'
snp_selection = 'ceu'
causal_variants = pd.read_csv('top_bottom_maf.csv')

parser = argparse.ArgumentParser(description='Process new true PRS.')
parser.add_argument('-variants', required=True)
parser.add_argument('-name', required=True)
args = parser.parse_args()

# select causal positions to be the top 1000 variants with the largest positive maf difference
causal_inds = np.array(causal_variants[args.variants])


if os.path.isfile(f"{outdir}sim{sim}/true_prs/prs_m_{m}_h2_{h2}_{args.name}.hdf5"):
    print(f"\nTrue PRS for iteration={sim} exists")
    print(f"If you would like to overwrite, remove {outdir}sim{sim}/true_prs/prs_m_{m}_h2_{h2}.hdf5")
else:
    print(f"\nSimulating true PRS for iteration={sim}")
    simulate_true_prs(args.name, causal_inds, m, h2, N_ADMIX, prefix=f"{outdir}sim{sim}/")

