#!/usr/bin/env Rscript
library(qqman)
library(RColorBrewer)
library(CMplot)
library(readxl)
CMplot(qyf11, plot.type=c("m","d"), multracks=TRUE,  
       threshold.lwd=c(1,2), threshold.col=c("black","grey"), amplify=TRUE,bin.size=1e6,ylim=c(0,20),
       chr.den.col=c("darkgreen", "yellow", "red"), signal.col=c("red","green"),signal.cex=c(1,1),
       file="pdf",file.output=TRUE,verbose=TRUE)
