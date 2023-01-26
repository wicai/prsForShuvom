## -----------------------------------------------------------------------------

# libraries
library(tidyverse)
library(ggplot2)
library(dplyr)



## -----------------------------------------------------------------------------

# read in the Allele Frequency (MAF) table for the European population
ceu <- read_tsv("output/sim1/maf/maf__ceu.txt") 
ceu_maf <- ceu %>% rename(maf_ceu=maf) %>% select(-population, -alleles, -X1) 

# read in the Minor Allele Frequency (MAF) table for the African population
yri <- read_tsv("output/sim1/maf/maf__yri.txt") 
yri_maf <- yri %>% rename(maf_yri=maf) %>% select(-population, -alleles, -X1)



## -----------------------------------------------------------------------------

# join the frames to get a wide frame with MAF in separate columns for each population
mafs <- full_join(ceu_maf, yri_maf, by=c('site', 'position'))

mafs <- mafs %>%
  mutate(
    maf_ceu = if_else(maf_ceu > .5, 1 - maf_ceu, maf_ceu),
    maf_yri = if_else(maf_yri > .5, 1 - maf_yri, maf_yri)
    )


## -----------------------------------------------------------------------------

# get the absolute difference between the CEU MAF and the YRI MAF as well as the ratio between both
diff_ratio_maf <- mafs %>%
  mutate(
    diff = maf_ceu - maf_yri,  #positive -> appears more often in CEU
    ratio = (maf_ceu + .01)/(maf_yri + .01) #>1 -> appears more often in CEU
    )




## -----------------------------------------------------------------------------

#top_n_variants <- function(df, n, col) {
#  top <- top_n(df, n, col)
#  return(top)
#}

top_1k_diff <- diff_ratio_maf %>% arrange(-diff) %>% head(1000)
bottom_1k_diff <- diff_ratio_maf %>% arrange(diff) %>% head(1000)

top_1k_ratio <- diff_ratio_maf %>% arrange(-ratio) %>% head(1000)
bottom_1k_ratio <- diff_ratio_maf %>% arrange(ratio) %>% head(1000)



## -----------------------------------------------------------------------------

export <- data.frame(top_1k_diff = top_1k_diff %>% pull(site),
                     bottom_1k_diff = bottom_1k_diff %>% pull(site),
                     top_1k_ratio = top_1k_ratio %>% pull(site),
                     bottom_1k_ratio = bottom_1k_ratio %>% pull(site)) 

write_csv(export, path = 'top_bottom_maf.csv')


