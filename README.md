# fair_sampling

## Running 

1. Run the original simulation to generate the true risk file for creating the MAF data and necessary directories: `python run_simulation.py --sim 1 --snp_selection ceu --snp_weighting ceu`

### Create the MAF data
1. Run this file to get all the MAF variants per population (instead of individually spaced variants): `python maf-2.py`
2. Run this R script to import the saved files from step 1 and create the CSV needed for the modified true risk: `Rscript maf-analysis.R`
3. Run this script for modified true risk in a separate terminal for each variant type with the following arguments: 

    - arg for variant type 1: `python modified_true_prs.py -variants top_1k_diff	-name top_diff`
    - arg for variant type 2: `python modified_true_prs.py -variants bottom_1k_diff -name bottom_diff`
   -  arg for variant type 3: `python modified_true_prs.py -variants top_1k_ratio -name top_ratio`
    - arg for variant type 4: `python modified_true_prs.py -variants bottom_1k_ratio -name bottom_ratio`


### Use the MAF data to run the modified simulation

1. Run the following scripts in a separate terminal session to create the MAF data:

- `python make_learning_curve_data.py -variant top_ratio`
- `python make_learning_curve_data.py -variant top_diff`
- `python make_learning_curve_data.py -variant bottom_diff`
- `python make_learning_curve_data.py -variant bottom_ratio`


### Create the learning curves
1. Once completed, change the variant type in cells 139 and 144 in this file to the variant you want to create the learning curve data for: https://github.com/stanford-policylab/fair_sampling/blob/main/emp_true_prs_prob.ipynb
2. Create the learning curves in this R notebook that imports the data from step 1 in this section: https://github.com/stanford-policylab/fair_sampling/blob/main/learning_curves.Rmd
