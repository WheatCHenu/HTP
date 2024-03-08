#!/usr/bin/env Rscript
library(tidyverse)
library(pheatmap)
library(ComplexHeatmap) 
library(circlize)
library(readr)
qyfside <- read.csv("side.csv", row.names=1, header = TRUE)
side <- pheatmap(qyfside,
                 # scale="column",
                 scale="row",
                 cluster_cols= F,
                 cluster_rows = T,
                 show_rownames = F,
                 show_colnames = F,
                 color = colorRampPalette(c("blue","yellow","red"))(11),
                 breaks = seq(-2, 2, 2)
                 ) 
