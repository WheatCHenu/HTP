#!/usr/bin/env Rscript
# install.packages("adegenet")
library(adegenet)
library(ape)
genotype_data <- read.table("1.txt", header = TRUE, sep = "\t")
# creat genind
genind_obj <- df2genind(genotype_data, sep = "\t")
# genind_obj <- df2genind(genotype_data, sep = "\t", col.ind = 1)
genetic_dist <- dist.genind(genind_obj)
genetic_dist <- dist(genind_obj)

# save genetic_distance_matrix file
write.table(as.matrix(genetic_dist), file = "genetic_distance_matrix.txt", sep = "\t")