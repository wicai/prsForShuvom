import simulation as SIM
from config import *
import argparse
import os

def main(sim,m,h2,weight,snp,p,r2,cases_yri,outdir,threads):
	# PART 0: CREATE DIRECTORIES NEEDED FOR SIMULATION
	for dirs in ["trees","admixed_data/input","admixed_data/output","true_prs","summary",
                "emp_prs/top_diff/cases", "emp_prs/top_diff/control",
                "emp_prs/bottom_diff/cases", "emp_prs/bottom_diff/control",
                "emp_prs/top_ratio/cases", "emp_prs/top_ratio/control",
                "emp_prs/bottom_ratio/cases", "emp_prs/bottom_ratio/control",
                "training/bottom_diff", "training/top_diff", "training/top_ratio", "training/bottom_ratio"]:
		os.system(f"mkdir -p {outdir}sim{sim}/{dirs}")
    
    for dirs in ["top_ratio", "bottom_ratio", "top_diff", "bottom_diff"]:
        os.system(f"mkdir -p learning_curve)data/{dirs}")
	
	# PART 1: SIMULATE POPULATIONS

	if os.path.isfile(f"{outdir}sim{sim}/trees/tree_all.hdf"):
		print(f"\nPopulation for iteration={sim} exists")
		print(f"If you would like to overwrite, remove {outdir}sim{sim}/trees/tree_all.hdf")
	else:
		print(f"\nSimulating populations for iteration={sim}")
		SIM.simulate_populations(N_CEU, N_YRI, N_MATE, N_ADMIX, rmap_file, prefix=f"{outdir}sim{sim}/")

	# PART 2: CONSTRUCT TRUE POLYGENIC RISK SCORES AND SPLIT INTO CASE/CONTROL

	if os.path.isfile(f"{outdir}sim{sim}/true_prs/prs_m_{m}_h2_{h2}.hdf5"):
		print(f"\nTrue PRS for iteration={sim} exists")
		print(f"If you would like to overwrite, remove {outdir}sim{sim}/true_prs/prs_m_{m}_h2_{h2}.hdf5")
	else:
		print(f"\nSimulating true PRS for iteration={sim}")
		SIM.simulate_true_prs(m, h2, N_ADMIX, prefix=f"{outdir}sim{sim}/")

	# PART 3: COMPUTE EMPIRICAL POLYGENIC RISK SCORES
	## optional parameters which can be modified: p-value, r2, weighting, snp-selection (future),
	## number of yri samples to be used as training
	SIM.create_emp_prs(m, h2, N_ADMIX, f"{outdir}sim{sim}/", p, r2,snp_weighting=weight,
						snp_selection=snp,num2decrease=cases_yri,num_threads=threads)

	# PART 4: SUMMARIZE RESULTS
	SIM.output_all_summary(sim,m,h2,f"{outdir}sim{sim}/", p, r2, snp_weighting=weight,
						   snp_selection=snp,num2decrease=cases_yri)


parser = argparse.ArgumentParser(description="Functions for simulating European, African, and Admixed populations and testing PRS building strategies for better generalization to diverse populations. Additional parameters can be adjusted in the config.py file")
parser.add_argument("--sim",help="Population identifier. Must give unique value if you want to run the simulation multiple times and retain outputs.", type=str, default="1")
parser.add_argument("--m",help="# of causal variants to assume", type=int, default=M)
parser.add_argument("--h2",help="heritability to assume", type=float, default=H2)
parser.add_argument("--snp_weighting",help="Weighting strategy for PRS building. Can use weights from European (ceu) or African (yri) GWAS as well as weights from a fixed-effects meta analysis of both GWAS (meta) or local-ancestry specific weights (la).", type=str, 
	choices={"ceu","yri","meta","la"}, default="ceu")
parser.add_argument("--snp_selection",help="SNP selection strategy for PRS building. Choice between using significant SNPs from a European (ceu) or African (yri) GWAS.", type=str, 
	choices={"ceu","yri","meta"}, default="ceu")
parser.add_argument("--pvalue",help="pvalue cutoff to be used for LD clumping", type=float, default=P)
parser.add_argument("--ld_r2",help="r2 cutoff to be used for LD clumping", type=float, default=R2)
parser.add_argument("--decrease_samples_yri",help="# of cases used in YRI analysis to represent the lack of non-European data", type=int, default=None)
parser.add_argument("--output_dir",help="location for output data to be written", type=str, default="output/")
parser.add_argument("--threads",help="# of threads to use for parallel processing", type=int, default=1)

args = parser.parse_args()
if args.output_dir[-1]!="/": args.output_dir+="/"
main(args.sim,args.m,args.h2,args.snp_weighting,
	args.snp_selection,args.pvalue,args.ld_r2,
	args.decrease_samples_yri,args.output_dir,args.threads)
