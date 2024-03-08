#!/usr/bin/env Rscript
##install.packages("Hmisc")
library(Hmisc)
##install.packages("corrplot")
library(corrplot)
##install.packages("PerformanceAnalytics")
library(PerformanceAnalytics)
##
f123 <- read.delim("HSI.txt", row.names=1, stringsAsFactors=TRUE)
cor_matr = cor(f123, method = 'pearson')
cor_matr
##
write.csv(cor(f123, method = 'pearson'),file="HSI.csv")
rcorr(as.matrix(f123))
symnum(cor_matr)
corrplot(cor_matr, type="upper", 
         order="hclust",
         tl.col ="black",
         hclust.method = 'ward',
         shade.col = "white",
         shade.lwd = 10,
         tl.cex= 0.4,
         addgrid.col = "white",
         diag=F)
corrplot(cor_matr, 
         type="upper", 
         order="hclust",
         tl.col ="black",
         tl.cex= 0.1,
         # addgrid.col = "white",
         # addCoef.col=F,
         diag=F)

