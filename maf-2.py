#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import simulation as SIM
from config import *
import argparse
import os


# In[2]:


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
from simulation.emp_risk import _load_data, _select_variants
from simulation.true_risk import return_diploid_genos
from config import *

vcf_file = "admixed_data/output/admix_afr_amer.query.vcf.gz"
path_tree_CEU="trees/tree_CEU_GWAS_nofilt.hdf"
path_tree_YRI="trees/tree_YRI_GWAS_nofilt.hdf"
snp_weighting="ceu"
snp_selection="ceu"
num2decrease=None
ld_distance=1e6
num_threads=8
prefix="output/sim1/"
    
# load all data
trees,sumstats,train_cases,train_controls,labels = _load_data(snp_weighting,snp_selection,
                                                                  path_tree_CEU,path_tree_YRI,
                                                                  prefix,M,H2,num2decrease)
# get simulation number
sim = prefix.split('sim')[1].split('/')[0]

# select PRS variants given above parameters
snps = _select_variants(sumstats[snp_selection],trees[snp_selection],M,H2,P,R2,snp_selection,prefix,ld_distance,num_threads,num2decrease)
var_list = snps.astype(int)


from multiprocessing import Manager, Pool
from functools import partial

def maf(pop, var):
    tree = trees[pop]
    
    site = var.site.id
    position = var.position
    alleles = var.alleles
    # convert phased haplotypes to genotype
    genos_diploid = return_diploid_genos(var.genotypes,tree)
    # calculate allele frequency of ALT allele
    freq = np.sum(genos_diploid,axis=1)/(2*genos_diploid.shape[1])
    table_row = [site, position, alleles, freq[0], pop]
    
    return table_row

def run_parallel():
    populations = ["yri", "ceu"]
    
    for pop in populations:
        print("pop: ", pop)
        p = Pool(25)
        func = partial(maf, pop)
        tree = trees[pop]
        variants = tree.variants()
        
        print("mapping ...")
        results = list(tqdm.tqdm(p.imap(func, variants), total=tree.num_sites))
        print("running ...")
        p.close()
        p.join()
        
        cols=['site', 'position', 'alleles', 'maf', "population"]
        df = pd.DataFrame(results, columns=cols)
        df.to_csv(f"{prefix}maf/maf__{pop}.txt",sep="\t")


run_parallel()


