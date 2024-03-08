#!/usr/bin/env Rscript
# setwd("")
library(RIdeogram)
karyotype <- read.table("167karyotype.txt", sep = "\t", header = T, stringsAsFactors = F)
Random_type <- read.delim("F:/167-g-1/GWAS/QTL/QTL??ͼ/Random_type.txt", stringsAsFactors=TRUE)

ideogram(karyotype = karyotype, label = Random_type, label_type = "marker",Lx = 135, Ly = 20)
convertSVG("chromosome.svg", device = "png")

library(RIdeogram)
gene_density <- read.delim("ckdrdrcQTL-HSI-density.txt", stringsAsFactors=TRUE)
Random_R <- read.delim("HSI-Random_type.txt", stringsAsFactors=TRUE)

ideogram(karyotype = karyotype,
         overlaid = gene_density,
         label = Random_R,
         label_type = "marker",
         # colorset1 = c("#33A02C", "#6a3d9a", "#ff7f00"), 
         Lx = 160, Ly = 5, output = "chromosome.svg")
convertSVG("chromosome.svg", device = "png")
convertSVG("chromosome.svg", device = "pdf")