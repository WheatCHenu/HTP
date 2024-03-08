#!/usr/bin/env Rscript
library("dplyr")
library("factoextra")
library(readxl)
# 
dir="http://210.75.224.110/github/MicrobiomeStatPlot/Data/Science2019/"
metadata <- read.delim("2.txt", row.names=1, stringsAsFactors=TRUE)
head(metadata, n = 3)
otutab <- read.delim("1.txt", row.names=1, stringsAsFactors=TRUE)
sub_metadata <- metadata[rownames(metadata) %in% colnames(otutab),]
count <- otutab[, rownames(sub_metadata)]
rcomp(t(count), scale. = TRUE)
## ?????µ???ʯeig(otu.pca, addlabels = TRUE))
ggsave(paste0("ZXJ.pca_screen.pdf"), p, width=89, height=56, units="mm")
ggsave(paste0("ZXJ.pca_screen.png"), p, width=89, height=56, units="mm")

#???????????var(otu.pca))

# ???Ʊ?��PCA???tu.pca, col.var = "contrib", gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07")))
ggsave(paste0("ZXJ.pca_cos.pdf"), p, width=89, height=89, units="mm")
ggsave(paste0("ZXJ.pca_cos.png"), p, width=89, height=89, units="mm")

#????��?????۲?ֵ???.pca))

# ????PCAͼ??????СΪcospca, pointsize = "cos2", pointshape = 21, fill = metadata$Group, repel = TRUE))
ggsave(paste0("ZXJ.pca_individuals.pdf"), p, width=89, height=56, units="mm")
ggsave(paste0("ZXJ.pca_individuals.png"), p, width=89, height=56, units="mm")

# ????PCAͼ??ֻ??ʾ?㣬?pca,
                   geom.ind = "point", # show points only ( not "text")
                   col.ind = metadata$Group, # color by groups
                   palette = c("#00AFBB", "#E7B800", "#FC4E07"),
                   addEllipses = TRUE, # Concentration ellipses
                   legend.title = "Groups"))
# ????ͼƬ??ָ??ͼƬΪpdple_group_ellipse.pdf"), p, width=89, height=56, units="mm")
ggsave(paste0("ZXJ.sample_group_ellipse.png"), p, width=89, height=56, units="mm")

